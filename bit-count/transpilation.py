import asyncio
import json, os
import random
import argparse
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit import qasm3
from qrmi.primitives import QRMIService
from qrmi.primitives.ibm import  get_target
BITLEN = 10

async def main():
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument(
        "--dir_path", 
        type=str, 
        required=True, 
        help="Path to the target directory"
    )
    
    args = parser.parse_args()
    work_root = Path(args.dir_path)
    #print(f"work_root: {work_root}")
    
    # Check if the directory exists
    #if work_root.is_dir():
    #    print(f"Valid directory provided: {work_root}")
    #else:
    #    print(f"Error: '{work_root}' is not a valid directory.")


    # 1. Initialize the IBM Quantum service and get the target backend
    #service = QiskitRuntimeService()
    service = QRMIService()
    #backend = service.backend("ibm_rensselaer")  # Replace with your specific backend

    resources = service.resources()
    if len(resources) == 0:
        raise ValueError("No quantum resource is available.")

    # Randomly select QR
    qrmi = resources[random.randrange(len(resources))]
    #print(qrmi.metadata())

    # Generate transpiler target from backend configuration & properties
    target = get_target(qrmi)

    # Create a PUB payload
    #target = await qrmi.get_target()
    qc_ghz = QuantumCircuit(BITLEN)
    qc_ghz.h(0)
    qc_ghz.cx(0, range(1, BITLEN))
    qc_ghz.measure_active()

    # 3. Generate the preset pass manager using the target
    pm = generate_preset_pass_manager(
        optimization_level=3,
        target=target,
        seed_transpiler=123,
    )
    # 4. Compile the circuit into IBM machine code (ISA)
    isa = pm.run(qc_ghz)

    # Extract shots
    #shots = options.get("shots", 10000) # default to 100000 if not set
    shots = 1000

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

    filename = str(work_root) + "/input.json"
    #filename = "/gpfs/u/home/QNTM/QNTMnkle/barn/rpi_qcsc_demo/bit-count/input.json"

    with open(filename, "w", encoding="utf-8") as primitive_input_file:
        json.dump(taskrunner_json, primitive_input_file, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
