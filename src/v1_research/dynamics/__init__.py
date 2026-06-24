"""Wilson-Cowan dynamics and rate solver backends."""

from v1_research.dynamics.jax_rk4 import JaxRK4Solver
from v1_research.dynamics.scipy_solver import ScipySolver
from v1_research.dynamics.solvers import RateResult, RateSolver, SolverConfig, make_solver, solve_rates
from v1_research.dynamics.transfer import (
    TransferConfig,
    TransferGrid,
    TransferTable,
    TransferTables,
    build_transfer_table,
    build_transfer_tables,
    integrate_siegert_kernel,
    siegert_kernel,
    siegert_rate,
)
from v1_research.dynamics.wilson_cowan import ExternalDrive, RateLayout, WilsonCowanEquation

__all__ = [
    "ExternalDrive",
    "JaxRK4Solver",
    "RateLayout",
    "RateResult",
    "RateSolver",
    "ScipySolver",
    "SolverConfig",
    "TransferConfig",
    "TransferGrid",
    "TransferTable",
    "TransferTables",
    "WilsonCowanEquation",
    "build_transfer_table",
    "build_transfer_tables",
    "integrate_siegert_kernel",
    "make_solver",
    "siegert_kernel",
    "siegert_rate",
    "solve_rates",
]
