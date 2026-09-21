# Create Your QCSC Workflow with Prefect

This hands-on tutorial guides you through building a small C++ program on a Slurm cluster and integrating it into a Prefect workflow using a custom `SlurmJobBlock` class.
On the Prefect workflow, we also use [Prefect Qiskit](https://github.com/qiskit-community/prefect-qiskit) to show how to write a complete QCSC workflow from scratch.

Our objective is to compute a count dictionary of sampler bitstrings using MPI programming on the QCSC architecture.

![Count BitStrings Flow](img-counts-flow.png)

## Prefect Core Concepts

We will use these terms:
- **Flow**: the end-to-end workflow defined in sampler_workflow_qrmi.py
- **Task**: individual steps inside the flow (e.g., runtime.sampler(...), counter.get(...))
- **Block**: reusable configuration + credentials stored in Prefect server
  - `qrmi-resource` : configuration/abstraction to access a specific quantum resource (QPU)
  - `bit-counter` : HPC job configuration (queue, nodes, executable path, modules)
  - `task-runner`: Quantum job configuration (executable, output) 
- **Variable**: run-time parameter stored server-side (sampler shots etc.)

## Create BitCounts Workflow
![BitCounts Setup Flow](img-counts-setup-flow.png)
## Step 1. Log in to the Slurm Cluster

Connect to the Slurm cluster login node using SSH. This is where we will develop the workflow.

<img src="icon-pc.png" alt="pc" width="50"/><br>
```bash
ssh login-node@slurm-cluster
```

## Step 2. Set up Python environment

Create a project directory ( our workdir is /data):

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
mkdir /data/bit-count && cd /data/bit-count
```

Create a virtual environment and activate:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
uv venv -p 3.12 && source .venv/bin/activate
```

Install necessary packages:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
uv pip install prefect-qiskit
uv pip install "git+ssh://git@github.com/shwetasalaria/qii-miyabi-kawasaki.git@bit-count#subdirectory=framework/prefect-slurm"
uv pip install qrmi
```

Check installations:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
uv pip list | grep -e "prefect" -e "qrmi"
```

You should see output like:

```text
prefect                   3.8.5
prefect-qiskit            0.2.1
prefect-slurm             0.1.0
qrmi                      0.24.4
```

## Step 3. Set up IBM Quantum System Access Credentials

Start the prefect server in background on the login node

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect server start --background --host 0.0.0.0
```

Make sure the virtual environment is activated

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
source .venv/bin/activate
```

Create a new Prefect profile:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect profile create bitcount
```

Prefect server is configured on the login node at port 4200. Set the config as:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Switch to the profile:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect profile use bitcount
```

In the project directory (`data/bit-count`), open a new Python file:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
vi qrmi_resource.py
```

Add following lines to the file:

```python
import os

from prefect.blocks.core import Block
from pydantic import Field, SecretStr

from qrmi import QuantumResource, ResourceType
from qrmi.primitives.ibm import get_target


class QRMIResource(Block):
    """
    Prefect block for configuring and accessing a QRMI resource.
    """

    _block_type_name = "QRMI Resource"
    _block_type_slug = "qrmi-resource"

    resource_id: str = Field(
        description="QRMI resource ID, e.g. test_heron"
    )

    endpoint: str = Field(
        description="IBM Quantum System endpoint"
    )

    iam_endpoint: str = Field(
        description="IBM IAM endpoint"
    )

    api_key: SecretStr = Field(
        description="IBM Quantum API key"
    )

    service_crn: SecretStr = Field(
        description="IBM Quantum service CRN"
    )

    async def get_target(self):
        """Get the Qiskit Target for the QRMI resource."""

        prefix = f"{self.resource_id}_QRMI_IBM_QS"

        os.environ[f"{prefix}_ENDPOINT"] = self.endpoint
        os.environ[f"{prefix}_IAM_ENDPOINT"] = self.iam_endpoint
        os.environ[f"{prefix}_IAM_APIKEY"] = self.api_key.get_secret_value()
        os.environ[f"{prefix}_SERVICE_CRN"] = self.service_crn.get_secret_value()

        resource = QuantumResource(
            self.resource_id,
            ResourceType.IBMQuantumSystem,
        )

        return get_target(resource)

```

Register the block schema from a file:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect block register -f qrmi_resource.py
```

Create `ibm-quantum-credentials` block instance:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect block create qrmi_resource
```
 
Follow the URL shown to configure the runtime block.
Specify the block name (which we can set as ibm-quantum-credentials) IBM Quantum backend name and credentials such as endpoint, IAM endpoint, API key and CRN.

![Setup Quantum Runtime](img-quantum-runtime-block.png)

Confirm you have access to the blocks you created:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect block ls
```

Example output:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ID                                   ┃ Type          ┃ Name                    ┃ Slug                                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 97791a8b-ca28-4dca-92a0-eb9ccb50a86d │ QRMI Resource │ ibm-quantum-credentials │ qrmi-resource/ibm-quantum-credentials │
└──────────────────────────────────────┴───────────────┴─────────────────────────┴───────────────────────────────────────┘

```

## Step 4. Create MPI Program

Maske sure you are in your your work directory:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
cd /data/bit-count
```

Open a C++ source code file:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
vi get_counts.cpp
```

Add following lines to the file:

```c++
#include <mpi.h>
#include <fstream>
#include <vector>
#include <iostream>

const uint32_t BITLEN = 10;
const uint32_t MAXVAL = 1 << BITLEN;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint32_t total_count = 0;
    std::vector<uint32_t> data;

    if (rank == 0) {
        std::ifstream fin("input.bin", std::ios::binary | std::ios::ate);
        total_count = fin.tellg() / sizeof(uint32_t);
        fin.seekg(0, std::ios::beg);

        data.resize(total_count);
        fin.read(reinterpret_cast<char*>(data.data()), total_count * sizeof(uint32_t));
        fin.close();
    }

    MPI_Bcast(&total_count, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    uint32_t local_n = total_count / size;
    std::vector<uint32_t> local_data(local_n);

    MPI_Scatter(rank == 0 ? data.data() : nullptr, local_n, MPI_UNSIGNED,
                local_data.data(), local_n, MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    std::vector<int> local_hist(MAXVAL, 0);
    for (auto v : local_data) {
        if (v < MAXVAL) local_hist[v]++;
    }

    std::vector<int> global_hist;
    if (rank == 0) global_hist.resize(MAXVAL, 0);

    MPI_Reduce(local_hist.data(),
               rank == 0 ? global_hist.data() : nullptr,
               MAXVAL, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream fout("output.json");
        fout << "{";
        bool first = true;
        for (uint32_t i = 0; i < MAXVAL; ++i) {
            if (global_hist[i] > 0) {
                if (!first) fout << ",";
                fout << "\"" << i << "\":" << global_hist[i];
                first = false;
            }
        }
        fout << "}\n";
        fout.close();
    }

    MPI_Finalize();
    return 0;
}
```

> [!NOTE]
> This program reads the `input.bin` file (download from this repo) including a 32-bit integer vector of bitstrings, splits the data across MPI processes, and counts how often each value appears.
> Each process builds a local histgram from its share of the data.
> MPI then combines all local results into a single global histogram, which rank 0 writes out as `output.json`.

Check the Open MPI library is loaded in your shell:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
mpirun --version
```

Make sure that MPI is available on all nodes in the cluster.

Since this program is lightweight, it's fine compiling on the login node:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
mpicxx -o get_counts get_counts.cpp
```

Check the output file:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
ls -l
```

Example output:

```text
-rwx------. 1 salaria salaria 130552 Jan  8 12:42 get_counts
-rw-------. 1 salaria salaria   1893 Jan  8 12:42 get_counts.cpp
```

Get the absolute path to the `get_counts` executable:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
realpath ./get_counts
```

Example output:

```text
/data/salaria/get_counts
```

We will need this path in the later step.

## Step 5. Configure Prefect block for MPI task

### Step 5.1 Prefect Block for MPI task

In this step, we define a Block Type (“bit-counter”) in Python.
This is the template/schema that tells Prefect what fields the block has and how it runs `get_counts` on the cluster.

In the project directory (`data/bit-count`), open a new Python file:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
vi get_counts_integration.py
```

Add following lines to the file:

```python
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
```

> [!NOTE]
> The `SlurmJobBlock` baseclass implements the mechanism to interact with the Slurm job scheduler.
> A subclass must implement the data input and output.
> The `get_inner` function is a trick to turn HPC job executions into Prefect Tasks. 

Register the block schema from a file:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect block register -f get_counts_integration.py
```

### Step 5.2 Create `bit-counter` block instance
In this step, we create a Block Instance (e.g., “bit-count”).
This is your environment-specific configuration, such as queue name, node count, and the executable path.
Create a new configuration for the Bit Counter block:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect block create bit-counter
```

This command will display a URL to the Prefect console.
Open it in your browser and fill in the following fields:

| Field | Value / Example |
|---|---|
| Block Name | `bit-count` |
| Root Directory | `/data/bit-count` <br> ( The absolute path to the `bit-count` directory) |
| Executable | `/data/bit-count/get_counts` <br> (The absolute path to the `get_counts` executable)|
| Executor | `sbatch`|
| Launcher | `srun`|
| Num Nodes | `2` |
| Num MPI Processes | `5` |

The other fields can be left blank.

Now this configuration is stored in the Prefect server and it can be used by many Prefect workflows.


## Step 6 Configure Prefect block for Quantum task

Now we write a Prefect block to run the quantum sampling task:

### Step 6.1 Define `task-runner` block type

In this step, we define a Block Type (“task-counter”) in Python.
This is the template/schema that tells Prefect what fields the block has and how it runs quantum job on quantum computer.

In the project directory (`/data/bit-count`), open a new Python file:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
vi get_task_runner.py
```

Add following lines to the file:

```python
import json
import numpy as np
from prefect import task
from prefect_slurm import SlurmJobBlock
from pydantic import Field

class TaskRunner(SlurmJobBlock):

    _block_type_name = "Task Runner"
    _block_type_slug = "task-runner"

    backend_name: str = Field(
        description="Backend name passed to task_runner as the first argument.",
        title="Backend Name",
    )

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
            job.backend_name,
            str(input_path),
            str(output_path),
        ]
        exit_code = await executor.execute_job(
            arguments=args,
            **job.get_job_variables(),
        )
```

> [!NOTE]
> The `SlurmJobBlock` baseclass implements the mechanism to interact with the Slurm job scheduler.
> The `TaskRunnerJobBlock` subclass defines backend, input and output.

Register the block schema from a file:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect block register -f get_task_runner.py
```

### Step 6.2 Create `task-runner` block instance

Create a new configuration for the Task Runner block:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect block create task-runner
```

Fill the fields (input, output, backend, executable) in the browser.

This configuration is stored in the Prefect server and it can be used by many Prefect workflows.

## Step 7. Create Prefect Workflow

Next, in the same directory, create a separate Python file to define the Prefect workflow:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
vi sampler_workflow_qrmi.py
```

Add following lines to the file:

```python
import asyncio
import json, os
from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.variables import Variable
from prefect_qiskit import QuantumRuntime
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from get_counts_integration import BitCounter
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit import qasm3
from get_task_runner import TaskRunner
from qiskit_ibm_runtime import RuntimeEncoder
from qiskit_ibm_runtime.decoders.result_decoder import ResultDecoder
from qrmi_resource import QRMIResource

BITLEN = 10

@flow(name="bit_count")
async def main():

    # Load configurations
    qrmi = await QRMIResource.load("ibm-quantum-credentials")
    counter = await BitCounter.load("bit-count")
    options = await Variable.get("bit-count")
    taskrunner = await TaskRunner.load("bit-count")

    # Create a PUB payload
    target = await qrmi.get_target()
    qc_ghz = QuantumCircuit(BITLEN)
    qc_ghz.h(0)
    qc_ghz.cx(0, range(1, BITLEN))
    qc_ghz.measure_active()


    pm = generate_preset_pass_manager(
        optimization_level=3,
        target=target,
        seed_transpiler=123,
    )
    isa = pm.run(qc_ghz)

    # Extract shots
    shots = options.get("shots", 100000) # default to 100000 if not set

    # Create input.json for task_runner
    coerced_pub = SamplerPub.coerce((isa,), shots=shots)

    # Generate OpenQASM3 string which can be consumed by IBM Quantum APIs
    qasm3_str = qasm3.dumps(
            coerced_pub.circuit,
            disable_constants=True,
            allow_aliasing=True,
            experimental=qasm3.ExperimentalFeatures.SWITCH_CASE_V1,
    )

    # Create SamplerV2 input
    input_json = {
    "pubs": [
        (qasm3_str, None, shots)
    ],
    "shots": shots,
    "options": {},
    "version": 2,
    "support_qiskit": False,
    }

    taskrunner_json = {"parameters": input_json, "program_id": "sampler"}

    filename = "/data/bit-count/input.json"
    with open(filename, "w", encoding="utf-8") as primitive_input_file:
        json.dump(taskrunner_json, primitive_input_file, indent=2)
        #json.dump(taskrunner_json, primitive_input_file, cls=RuntimeEncoder, indent=2)

    # Quantum execution
    result = await taskrunner.run()

    # Read output
    with open('/data/bit-count/output.json', 'r') as f:
        results = ResultDecoder.decode(f.read())

    # MPI execution
    samples = results["results"][0]["data"]["meas"]["samples"]
    num_bits = results["results"][0]["data"]["meas"]["num_bits"]
    bitstrings = [format(int(s, 16), f"0{num_bits}b") for s in samples]
    counts = await counter.get(bitstrings)

    # Save in Prefect artifacts
    await create_table_artifact(
        table=[list(counts.keys()), list(counts.values())],
        key="sampler-count-dict",
     )


if __name__ == "__main__":
    asyncio.run(main())
```

## Step 8. Execute the workflow

Make sure the Quantum Runtime block exist:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect block inspect qrmi-resource/ibm-quantum-credentials
```

The output may look like:

```text

┌───────────────────┬──────────────────────────────────────────────────┐
│ Block Type        │ QRMI Resource                                    │
│ Block id          │ 97791a8b-ca28-4dca-92a0-eb9ccb50a86d             │
├───────────────────┼──────────────────────────────────────────────────┤
│ resource_id       │ test_heron                                       │
│ endpoint          │ http://172.16.17.118:8080                        │
│ iam_endpoint      │ https://iam.test.cloud.ibm.com                   │
│ api_key           │ ********                                         │
│ service_crn       │ ********                                         │
└───────────────────┴──────────────────────────────────────────────────┘
```

Set the sampler options for the IBM Qiskit Runtime API:

<img src="icon-slurm.png" alt="slurm" width="50"/><br>
```bash
prefect variable set bit-count '{"options": {"shots": 100000}}' --overwrite
```

Verify the variable:

<img src="icon-slurm.png" alt="mdx" width="50"/><br>
```bash
prefect variable inspect bit-count
```

Example output:

```text
Variable(
    id='c28e5b6c-2f5d-4ad7-82de-3e8f61e765d7',
    created=DateTime(2026, 9, 10, 2, 14, 40, 169461,
tzinfo=Timezone('UTC')),
    updated=DateTime(2026, 9, 14, 20, 18, 51, 25000,
tzinfo=Timezone('UTC')),
    name='bit-count',
    value={'options': {'shots': 100000}},
    tags=[]
)
```

To execute the workflow, run the following Python script:

<img src="./images/icon-slurm.png" alt="mdx" width="50"/><br>
```bash
python sampler_workflow_qrmi.py
```

We can also monitor the progress on the Prefect console:

![Get Counts Flow Run](./images/img-prefect-slurm.png)

Upon successful completion of the workflow, Prefect will generate the following artifacts:

- `sampler-count-dict`: Count dictionary computed by our MPI program.
- `job-metrics`: Performance metrics of IBM primitive execution.
- `slurm-job-metrics`: Performance metrics of Slurm job execution.

See the official [Artifacts](https://docs.prefect.io/v3/concepts/artifacts) guide about Prefect artifacts.
Example data is available in [here](./examples/create_qcsc_workflow/).

The metrics artifacts are automatically generated by Prefect integration libraries.
This information might be useful to optimize computing resources.

> [!NOTE]
> Note that this example does not significantly benefit from MPI parallel execution,
> as data input and output on the rank 0 process is the dominant performance bottleneck.
> This example is chosen to demonstrate how MPI programs look like.

---
*END OF TUTORIAL*
