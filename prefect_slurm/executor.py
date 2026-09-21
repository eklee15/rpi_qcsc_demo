"""Job executor implementations."""

import asyncio
import json
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Self

from jinja2 import Environment, PackageLoader
from prefect import task
from prefect.artifacts import create_table_artifact
from prefect.exceptions import MissingContextError
from prefect.logging import get_logger, get_run_logger

from .utils import run_command


class JobExecutorBase(ABC):
    """
    Job Executor base class.

    This base class defines a standardized `execute_job` method for job submission workflows.
    The method internally generates a job script using a built-in template and submits
    it to the job scheduler daemon.

    Once the job is scheduled, execution proceeds in a non-blocking manner. The method
    awaits job completion and returns the job's exit code upon termination.

    The class also behaves as a context manager which has a responsibility of
    setting up and cleaning up the environment before and after job submission.
    """

    JINJA_ENV: ClassVar[Environment] = Environment(loader=PackageLoader("prefect_slurm", "templates"))
    TEMP_NAME: ClassVar[str] = ""
    MAX_LOG_SIZE: ClassVar[int] = 10_000

    def __init__(
        self,
        root_dir: Path,
    ):
        """Instantiate new job executor.

        Args:
            root_dir: A directory where the job executor create a sub directory for job data generation.
        """
        self._root_dir = root_dir
        self._work_dir: Path | None = None

    @property
    def work_dir(self) -> Path:
        if self._work_dir is None:
            raise RuntimeError("Working directory is not set.")
        return self._work_dir

    @property
    def logger(self):
        try:
            return get_run_logger()
        except MissingContextError:
            return get_logger(self.__class__.__name__)

    def truncate_log(self, text: str) -> str:
        """Truncate very long logs.

        Logging a long text can result in 422 Unprocessable Entity from Prefect Cloud.
        This function truncates a single log text when it exceeds the limit length.

        Args:
            text: A text to be logged.

        Returns:
            Truncated log.
        """
        if len(text) > self.MAX_LOG_SIZE:
            text = text[: self.MAX_LOG_SIZE] + f"... (truncated {len(text) - self.MAX_LOG_SIZE} chars)"
        return text

    def __enter__(self) -> Self:
        if self._work_dir is not None:
            raise RuntimeError(
                "This executor instance is already in the executor context. Executor context cannot be nested."
            )
        self._work_dir = Path(tempfile.mkdtemp(prefix="job_", dir=self._root_dir))
        self.logger.debug(f"Created a temporary work directory at '{self._work_dir}'.")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is not None:
            self.logger.warning(
                f"The {exc_type.__name__} exception occured during job execution. "
                f"Investigate the work directory at '{self.work_dir}'."
            )
        else:
            shutil.rmtree(self.work_dir)
            self.logger.debug(f"Removed a temporary work directory at '{self._work_dir}'.")
            self._work_dir = None

    def write_job_script(self, **kwargs) -> str:
        """Write job script in the work directory.

        .. note::
            This method is expected called inside the :meth:`execute_job` method.
            A developer who writes a custom job block doesn't need to explicitly call this method.
            An executor subclass may overwrite this method to use a custom template,
            or may hook to format keyword arguments before consumed by the template.

        .. note::
            This method assumes a template name with extension `.j2`,
            which must be specified by the class variable :attribute:`TEMP_NAME`.

        Args:
            kwargs: Variables to feed into a job script template.

        Returns:
            An absolute path to the job script.
        """
        if self.work_dir is None:
            raise RuntimeError(
                "An executor method is called outside of an executor context. Try calling within an executor context."
            )
        script = self.JINJA_ENV.get_template(self.TEMP_NAME).render(
            work_dir=str(self.work_dir),
            **kwargs,
        )
        with open(self.work_dir / self.TEMP_NAME.rstrip(".j2"), mode="w") as file:
            file.write(script)
        return file.name

    @abstractmethod
    async def execute_job(
        self,
        timeout: float | None = None,
        **kwargs,
    ) -> int:
        """Run job in the work directory.


        Args:
            timeout: Amount of time in seconds for waiting the job to complete.
            kwargs: Variables to feed into a job script template.

        Raises:
            TimeoutError: When job doesn't finish within the timeout.

        Returns:
            Exit code of the job.
        """
        ...


class LocalExecutor(JobExecutorBase):
    """
    Local Program Executor.

    This executor runs a given executable directly as a shell script in the local environment.
    Standard output and error streams are forwarded to the Prefect server, but no performance
    metrics or artifacts are generated.

    It is particularly useful for validating code and workflows locally before submitting
    jobs to the Slurm cluster.
    """

    TEMP_NAME = "local.sh.j2"

    @task(
        name="run_local",
        tags=["res: local"],
    )
    async def execute_job(
        self,
        timeout: float | None = None,
        **kwargs,
    ) -> int:
        proc = await asyncio.create_subprocess_exec(
            "sh",
            self.write_job_script(**kwargs),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            text=False,
        )
        self.logger.info(f"Created a local process with PID {proc.pid}")

        try:
            async with asyncio.timeout(delay=timeout):
                stdout, stderr = await proc.communicate()
        except TimeoutError as ex:
            proc.kill()
            raise TimeoutError(f"Process {proc.pid} timeout expired.") from ex

        if logs := stdout.decode("utf-8", errors="replace"):
            logs = self.truncate_log(logs)
            self.logger.info(logs)

        if logs := stderr.decode("utf-8", errors="replace"):
            logs = self.truncate_log(logs)
            self.logger.error(logs)

        if proc.returncode is None:
            return -1
        return proc.returncode

class BatchExecutor(JobExecutorBase):
    """
    SLURM Batch Scheduler Executor.

    This executor submits jobs to the SLURM scheduler and monitors their execution.
    After job completion, it extracts final job metrics and stores them in the Prefect server
    as a table artifact under the data key `slurm-job-metrics`.

    This executor must be run in an environment where the SLURM daemon is active
    and the `sbatch` command is available.
    """

    TEMP_NAME = "batch.slurm.j2"

    async def _wait(
        self,
        job_id: str,
        watch_poll_interval: float = 5.0,
    ) -> dict[str, Any]:
        """Wait until job completes and retrieve final status."""
        try:
            while True:
                stdout = await run_command(
                    "scontrol", "show", "job", "-o", job_id
                )

                if stdout.strip():
                    # scontrol -o gives a single line of space-separated key=value pairs per job
                    for line in stdout.strip().split('\n'):
                        # Parse key-value pairs (handling potential spaces inside values if any,
                        # though -o usually formats tightly)
                        job_data = {}
                        for token in line.split():
                            if '=' in token:
                                k, v = token.split('=', 1)
                                job_data[k] = v

                        # Map scontrol fields to your required dictionary keys
                        # Note: scontrol uses 'JobId', 'JobState', 'ExitCode', 'RunTime', 'NumCPUs', 'NodeList'
                        out = {
                            'JobID': job_data.get('JobId', job_id),
                            'State': job_data.get('JobState', 'UNKNOWN'),
                            'ExitCode': job_data.get('ExitCode', '0:0'),
                            'Elapsed': job_data.get('RunTime', '0:00'),
                            'AllocCPUS': job_data.get('NumCPUs', job_data.get('AllocCPUS', '0')),
                            'NodeList': job_data.get('NodeList', ''),
                        }

                        # Check if job finished
                        final_states = ['BOOT_FAIL', 'COMPLETED', 'FAILED', 'CANCELLED', 'DEADLINE', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED']
                        if out['State'] in final_states:
                            exit_code = out['ExitCode'].split(':')[0]
                            self.logger.info(f"Job {job_id} finished with state {out['State']} and exit code {exit_code}")
                            return out

                stdout = await run_command(
                    "sacct", "-j", job_id,
                    "--format=JobID,State,ExitCode,Elapsed,AllocCPUS,NodeList",
                    "--parsable2", "--noheader"
                )

                if stdout.strip():
                    for line in stdout.strip().split('\n'):
                        fields = line.split('|')
                        job_id_field = fields[0]

                        # Skip job steps (.batch, .extern)
                        if '.' in job_id_field or '+' in job_id_field:
                            continue

                        out = {
                            'JobID': fields[0],
                            'State': fields[1],
                            'ExitCode': fields[2],
                            'Elapsed': fields[3],
                            'AllocCPUS': fields[4],
                            'NodeList': fields[5],
                        }

                        # Check if job finished
                        final_states = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL']
                        if out['State'] in final_states:
                            exit_code = out['ExitCode'].split(':')[0]
                            self.logger.info(f"Job {job_id} finished with state {out['State']} and exit code {exit_code}")
                            return out

                await asyncio.sleep(watch_poll_interval)

        except asyncio.CancelledError:
            await run_command("scancel", job_id)
            self.logger.warning(f"Job {job_id} was cancelled.")
            return {}

    def _create_job_artifact(
        self,
        job_id: str,
        job_status: dict[str, str],
    ) -> dict[str, Any]:
        """Convert job status output into Prefect artifact data."""
        exit_code, signal = job_status.get('ExitCode', '0:0').split(':')

        return {
            "job_id": job_id,
            "state": job_status.get("State"),
            "exit_code": exit_code,
            "signal": signal,
            "elapsed_time": job_status.get("Elapsed"),
            "allocated_cpus": job_status.get("AllocCPUS"),
            "node_list": job_status.get("NodeList"),
        }

    @task(
        name="run_slurm",
        tags=["res: slurm"],
    )
    async def execute_job(
        self,
        timeout: float | None = None,
        **kwargs,
    ) -> int:
        job_id = await run_command("sbatch", "--parsable", self.write_job_script(**kwargs))
        job_id = job_id.strip()

        try:
            async with asyncio.timeout(delay=timeout):
                final_status = await self._wait(job_id=job_id)
        except TimeoutError as ex:
            raise TimeoutError(f"Job {job_id} timeout expired.") from ex

        if final_status:
            # Read output files
            self.logger.info("READ OUTPUT FILES")
            stdout_file = os.path.join(self.work_dir, f"output.out")
            if os.path.exists(stdout_file):
                with open(stdout_file) as f:
                    logs = self.truncate_log(f.read())
                    if logs:
                        self.logger.info(logs)

            stderr_file = os.path.join(self.work_dir, f"output.err")
            if os.path.exists(stderr_file):
                with open(stderr_file) as f:
                    logs = self.truncate_log(f.read())
                    if logs:
                        self.logger.error(logs)

            # Create artifact
            art_dict = self._create_job_artifact(job_id=job_id, job_status=final_status)
            await create_table_artifact(
                table=[list(art_dict.keys()), list(art_dict.values())],
                key="slurm-job-metrics",
            )

        exit_code = final_status.get('ExitCode', '-1:0').split(':')[0]
        return int(exit_code)
