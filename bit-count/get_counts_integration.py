import json
import numpy as np
from prefect import task
from prefect_slurm import SlurmJobBlock

BITLEN = 10

class BitCounter(SlurmJobBlock):

    _block_type_name = "Bit Counter"
    _block_type_slug = "bit-counter"

    async def get(
        self,
        bitstrings: list[str],
    ) -> dict[str, int]:
        return await get_inner(self, bitstrings)

@task(name="get_counts_mpi")
async def get_inner(
    job: BitCounter,
    bitstrings: list[str],
) -> dict[str, int]:
    with job.get_executor() as executor:
        # Write file
        u32int_array = np.array(
            [int(b, base=2) for b in bitstrings],
            dtype=np.uint32,
        )
        u32int_array.tofile(executor.work_dir / "input.bin")
        # Run MPI program
        exit_code = await executor.execute_job(**job.get_job_variables())
        assert exit_code == 0

        # Read file
        with open(executor.work_dir / "output.json", "r") as f:
            int_counts = json.load(f)

        return {
            format(int(k), f"0{BITLEN}b"): v
            for k, v in int_counts.items()
        }
