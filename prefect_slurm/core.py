"""Base block document for HPC job configuration."""

from pathlib import Path
from typing import Literal

from prefect.blocks.core import Block
from pydantic import Field

from .executor import BatchExecutor, JobExecutorBase, LocalExecutor


class SlurmJobBlock(Block):
    """Block document including fundamental setup for job execution."""

    _block_type_name = "Slurm Job"
    _block_type_slug = "slurm_job"

    work_root: str = Field(
        description=(
            "Root directory to store temporary files for program execution. Usually a sub-directory is created per job."
        ),
        title="Root Directory",
    )

    executable: str = Field(
        description="Absolute path of the executable to run in the job. Arguments must be separately specified.",
        title="Executable",
    )

    executor: Literal[
        "sbatch",
        "local",
    ] = Field(
        default="local",
        description="A type of job scheduler to run the executable.",
        title="Executor",
    )

    launcher: Literal[
        "single",
        "srun",
        "mpirun",
    ] = Field(
        default="single",
        description=(
            "A command to control program execution environment such as MPI. "
            "When 'single' is selected, the executable is directly executed in the job."
        ),
        title="Launcher",
    )

    partition: str | None  = Field(
        default="compute",
        description=(
            "The name of parition to consume token from at job execution. "
            "This field is required for execution in the Slurm environment."
        ),
        title="Partition",
    )

    qpu: str | None = Field(
    default=None,
    description=(
        "Target QPU name or comma-separated list of QPU names for execution "
        "(e.g., 'x' or 'x,y,z'). If not specified, uses the default QPU."
    ),
    title="QPU",
    )

    num_nodes: int | None = Field(
        default=None,
        description=(
            "Number of compute node to request. Maximum number of available node per job depends on the queue name."
        ),
        gt=0,
        title="Num Nodes",
    )

    mpiprocs: int | None = Field(
        default=None,
        description="Number of MPI process per node to request.",
        gt=0,
        title="Num MPI Processes",
    )

    mpi_options: list[str] | None = Field(
        default=None,
        description="Execution options for the MPI launcher.",
        title="MPI Options",
    )

    ompthreads: int | None = Field(
        default=None,
        description="Number of OpenMP threads to request.",
        gt=0,
        title="Num OMP Threads",
    )

    walltime: str | None = Field(
        default=None,
        description=(
            "Limit for elapse time in the format '[[hour:]minute:]second'. "
            "This helps the scheduler to backfill the empty slots with your job."
        ),
        title="Max Walltime",
    )

    modules: list[str] | None = Field(
        default=None,
        description=(
            "List of modules to load within the job script. "
            "Intel OneAPI (intel and impi) is automatically loaded at the batch job execution."
        ),
        title="Load Modules",
    )

    environments: dict[str, str | int] | None = Field(
        default=None,
        description="Dictionary of environment variables to expose in the job script.",
        title="Environment Variables",
    )

    def get_executor(self) -> JobExecutorBase:
        """Get proper job executor class according to the document.

        Returns:
            A class that implements the JobExecutor protocol.
        """
        match self.executor:
            case "local":
                return LocalExecutor(root_dir=Path(self.work_root))
            case "sbatch":
                return BatchExecutor(root_dir=Path(self.work_root))

    def get_job_variables(self) -> dict:
        """Generate dictionary of variables to feed into job script template.

        Returns:
            A dictionary of job script variables.
        """
        return self.model_dump(
            mode="python",
            include=[
                "executable",
                "launcher",
                "partition",
                "qpu",
                "num_nodes",
                "mpiprocs",
                "mpi_options",
                "ompthreads",
                "walltime",
                "modules",
                "environments",
            ],
            exclude_none=True,
        )
