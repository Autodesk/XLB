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
        self.ids = {}
        self.next_id = 0

    def register_boundary_condition(self, boundary_condition):
        """
        Register a boundary condition.
        """
        id = self.next_id
        self.next_id += 1
        self.ids[boundary_condition] = id
        return id

boundary_condition_registry = BoundaryConditionRegistry()
