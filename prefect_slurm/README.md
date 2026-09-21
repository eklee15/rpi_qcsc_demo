# Prefect Slurm Integration

This package provides a Prefect integration for Slurm based environment.

The core object that the package provides is the `SlurmJobBlock` block, which stores configuration for your HPC job.
This configuration can be provided either through code or via the GUI of the Prefect console.

## Getting Started

To start the Prefect server and register the schema of the block, run the following command from the terminal:

```bash
uv pip install .
```

Start a local Prefect server and update the endpoint URL:

```bash
prefect server start --background
prefect config set PREFECT_API_URL=http://localhost:4200/api
```

Register the schema in our integration:

```bash
prefect block register -m prefect_slurm
```

Once the block is registered, you can find the `SlurmJobBlock` in the Prefect console.
This block can submit a job to the SLURM scheduler deployed on a cluster, as well as execute jobs in the local shell for code validation.
A target executable can also be launched inside the MPI environment using `mpirun`.

## Example

By default, this block can only call a command available in the current shell as a Python subprocess.
You can also specify the absolute path to run any executable you may manually build.
For example:

```python
import asyncio
from prefect import flow
from prefect_slurm import SlurmJobBlock

@flow
async def say_hello():

    # Define an HPC job
    block = SlurmJobBlock(
        work_root="/path/to/work/directory",
        executable="echo",
        executor="sbatch",
        launcher="single",
        queue_name="debug-c",
        project="group1",
        num_nodes=1,
    )

    # Execute with provided settings
    with block.get_executor() as executor:
        await executor.execute_job(
            arguments=["Hello Slurm Cluster!"],
            **block.get_job_variables(),
        )

if __name__ == "__main__":
    asyncio.run(say_hello())
```

This script executes `echo Hello Slurm Cluster!` on a compute node of the cluster.
The standard output is forwarded to the Prefect server, so you can check the output string on the Prefect console.

To run the code locally for validation, you may use the following settings instead:

```python
block = SlurmJobBlock(
    work_root="/path/to/work/directory",
    executable="echo",
    executor="local",
    launcher="single",
)
```

This setting can also be configured on the Prefect console GUI.
Then, you can load a particular setting in the workflow code:

```python
block = await SlurmJobBlock.load("say_hello_conf")
```

This pattern allows you to decouple execution settings from the codebase,
which improves the reusability of your workflow code.
For example, you can switch from the local executor to the Slurm executor without modifying the code.

## Custom Job Block

The `SlurmJobBlock` alone rarely provides practical benefits.
In a realistic workflow, you may need to prepare data before job submission and format the computed outcome.
We recommend developers subclass `SlurmJobBlock` to implement such pre-processing and post-processing for APIs of target executables.

This package also provides the `PyFunctionJob` block as a reference implementation.
This class inherits from `SlurmJobBlock` and implements a `.run()` method to execute arbitrary Python functions on any cluster via serialization with `cloudpickle`.

```python
import asyncio
import numpy as np
from prefect import flow
from prefect_slurm.pyfunc import PyFunctionJob

def numpy_fn(arg1: np.ndarray, arg2: np.ndarray) -> np.ndarray:
    return arg1 + arg2

@flow
async def pyfunc_numpy(tmp_path):
    rng = np.random.default_rng(123)

    # Load the settings from Prefect server
    func_job = await PyFunctionJob.load("config1")

    vec1 = rng.random(100)
    vec2 = rng.random(100)

    # Run Python function as a job and get the return value
    job_ret = await func_job.run(numpy_fn, arg1=vec1, arg2=vec2)

if __name__ == "__main__":
    asyncio.run(pyfunc_numpy())
```

## Contribution Guidelines

We avoid adding `SlurmJobBlock` subclasses for every scientific computing library unless many algorithm sub-packages depend on them.
Project-specific subclasses should be bundled with the algorithm sub-package itself.

Any code changes to this package must pass the `pre-commit` check.
All new features must be tested, and corresponding tests should be added to `./tests/unittest`.
We use `pytest` as the testing framework.
Breaking API change is not allowed unless you update all downstream sub-packages in this repository.
