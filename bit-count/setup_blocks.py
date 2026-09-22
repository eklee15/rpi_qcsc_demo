"""Provision the Prefect Blocks and Variable used by sampler_workflow_qrmi.py.
Run with: python setup_blocks.py
"""
import asyncio

from prefect.blocks.core import Block
from prefect.variables import Variable

from get_counts_integration import BitCounter
from get_task_runner import TaskRunner

WORK_ROOT = "/gpfs/u/home/QNTM/QNTMnkle/barn/rpi_qcsc_demo/bit-count"
QPU = "ibm_rensselaer"
PARTITION = "quantum"

BIT_COUNTER_BLOCK_NAME = "bit-count"
BIT_COUNTER_CONFIG = dict(
    work_root=WORK_ROOT,
    executable=f"{WORK_ROOT}/get_counts",
    executor="sbatch",
    launcher="mpirun",
    partition=PARTITION,
    qpu=QPU,
    num_nodes=1,
    mpiprocs=1,
    ompthreads=1,
    walltime="00:20:00",
)

TASK_RUNNER_BLOCK_NAME = "bit-count"
TASK_RUNNER_CONFIG = dict(
    work_root=WORK_ROOT,
    executable="task_runner",
    executor="sbatch",
    launcher="srun",
    partition=PARTITION,
    qpu=QPU,
    num_nodes=1,
    mpiprocs=1,
    walltime="00:20:00",
    input_file=f"{WORK_ROOT}/input.json",
    output_file=f"{WORK_ROOT}/output.json",
)

BIT_COUNT_VARIABLE_NAME = "bit-count"
BIT_COUNT_VARIABLE_VALUE = {
    "options": {
        "shots": 1000,
    },
}

async def save_block(block: Block, name: str) -> None:
    await block.save(name, overwrite=True)
    print(f"Saved block '{block.get_block_type_slug()}/{name}'")


async def save_variable(name: str, value: dict) -> None:
    await Variable.aset(name, value, overwrite=True)
    print(f"Saved variable '{name}'")


async def main() -> None:
    await save_block(BitCounter(**BIT_COUNTER_CONFIG), BIT_COUNTER_BLOCK_NAME)
    await save_block(TaskRunner(**TASK_RUNNER_CONFIG), TASK_RUNNER_BLOCK_NAME)
    await save_variable(BIT_COUNT_VARIABLE_NAME, BIT_COUNT_VARIABLE_VALUE)


if __name__ == "__main__":
    asyncio.run(main())
