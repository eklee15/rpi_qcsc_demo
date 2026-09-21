import json
import numpy as np
from prefect import task
from prefect_slurm import SlurmJobBlock
from pydantic import Field

class TaskRunner(SlurmJobBlock):

    _block_type_name = "Task Runner"
    _block_type_slug = "task-runner"

    input_file: str = Field(
        default="input.json",
        description="Input JSON file name.",
        title="Input JSON File",
    )

    output_file: str = Field(
        default="output.json",
        description="Output JSON file name.",
        title="Output JSON File",
    )

    async def run(self):
        return await run_inner(self)

@task(name="run_task_runner")
async def run_inner(job: TaskRunner):
    with job.get_executor() as executor:
        input_path = executor.work_dir / job.input_file
        output_path = executor.work_dir / job.output_file

        args: List[str] = [
            job.qpu,
            str(input_path),
            str(output_path),
        ]
        exit_code = await executor.execute_job(
            arguments=args,
            **job.get_job_variables(),
        )
