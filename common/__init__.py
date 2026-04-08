"""PCB thermal modeling - shared physics and utilities."""

from .pcb_physics import (
    PCBDomain,
    HeaterPatch,
    DirichletBC,
    EdgeDirichletBC,
    RadiativeBC,
    initial_condition,
    pcb_rhs,
    get_node_coordinates,
)

__all__ = [
    "PCBDomain",
    "HeaterPatch",
    "DirichletBC",
    "EdgeDirichletBC",
    "RadiativeBC",
    "initial_condition",
    "pcb_rhs",
    "get_node_coordinates",
]
