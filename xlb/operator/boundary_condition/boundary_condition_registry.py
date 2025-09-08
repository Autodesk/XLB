"""
Registry for boundary conditions in a LBM simulation.
"""


class BoundaryConditionRegistry:
    """
    Registry for boundary conditions in a LBM simulation.
    """

    def __init__(
        self,
    ):
        self.id_to_bc = {}  # Maps id number to boundary condition
        self.bc_to_id = {}  # Maps boundary condition to id number
        self.next_id = 1  # 0 is reserved for no boundary condition

    def register_boundary_condition(self, boundary_condition):
        """
        Register a boundary condition.
        """
        _id = self.next_id
        self.next_id += 1
        self.id_to_bc[_id] = boundary_condition
        self.bc_to_id[boundary_condition] = _id
        print(f"registered bc {boundary_condition} with id {_id}")
        return _id


boundary_condition_registry = BoundaryConditionRegistry()
