"""Reference implementation for Python function execution."""

from collections.abc import Callable
from typing import Any

import cloudpickle as pickle
from jinja2 import Environment, PackageLoader

from .core import SlurmJobBlock

JINJA_ENV = Environment(loader=PackageLoader("prefect_slurm", "templates"))


class PyFunctionJob(SlurmJobBlock):
    """
    Python Function Executor for SLURM cluster via Prefect Block Integration.

    This class enables the execution of arbitrary Python functions on the Slurm cluster
    by submitting jobs to its scheduler using `cloudpickle`. The function and its arguments
    are serialized and stored in the working directory, along with an auto-generated Python
    script that executes the pickled function. The function's return value is also serialized
    and retrieved by the executor.

    This approach is ideal for offloading computationally intensive Python workloads
    to the HPC environment.

    Example usage:

        import asyncio
        from prefect_slurm.pyfunc import PyFunctionJob

        async def main():
            # Load the Prefect block configuration named 'pyfunc_setup'
            executor = await PyFunctionJob.load("pyfunc_setup")

            def my_function(name: str):
                return f"Hello {name}"

            result = await executor.run(my_function, "cluster")
            print(result)

        if __name__ == "__main__":
            asyncio.run(main())

    To use this executor, define and save a Prefect block document for `PyFunctionJob`
    in the Prefect console under the name `pyfunc_setup`. Alternatively, you can
    instantiate the block directly in your script with full configuration parameters.
    """

    _block_type_name = "PyFunction Job"
    _block_type_slug = "pyfunction_job"

    async def run(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Run Python function in the HPC environment.

        Args:
            func: Python function to execute.
            args: Arguments to call the Python function.
            kwargs: Keyword arguments to call the Python function.

        Raises:
            RuntimeError: When a job returns nonzero exit code.

        Returns:
            A return value from the Python function.
        """
        with self.get_executor() as executor:
            # Serialize the function and input data and create a Python script.
            with open(executor.work_dir / "source.pkl", mode="wb") as source_fp:
                pickle.dump((func, args, kwargs), source_fp)

            result_file = executor.work_dir / "result.pkl"

            py_script = JINJA_ENV.get_template("run_py.j2").render(
                source_file=source_fp.name,
                result_file=str(result_file),
            )
            with open(executor.work_dir / "py_script", mode="w") as pyscript_file:
                pyscript_file.write(py_script)

            # Run on HPC
            exit_code = await executor.execute_job(
                arguments=[pyscript_file.name],
                **self.get_job_variables(),
            )

            if exit_code != 0:
                raise RuntimeError(f"{self.__class__.__name__} exits with nonzero exit code. Terminating the process.")

            # Read result
            with open(result_file, mode="rb") as result_fp:
                result = pickle.load(result_fp)

        return result
