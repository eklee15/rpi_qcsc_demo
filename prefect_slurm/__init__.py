"""Integration of Prefect for SLURM cluster"""

from .core import SlurmJobBlock
from .pyfunc import PyFunctionJob

__all__ = [
    "SlurmJobBlock",
    "PyFunctionJob",
]
