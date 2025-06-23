from typing import Any
import copy

import warp as wp

from xlb.operator.operator import Operator
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_masker import IndicesBoundaryMasker
import neon


class MultiresIndicesBoundaryMasker(IndicesBoundaryMasker):
    """
    Operator for creating a boundary mask using indices of boundary conditions in a multi-resolution setting.
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.WARP,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)
        if self.compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {self.compute_backend} backend.")

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional_dict, _ = self._construct_warp()
        functional_domain_bounds = functional_dict.get("functional_domain_bounds")
        functional_interior_bc_mask = functional_dict.get("functional_interior_bc_mask")
        functional_interior_missing_mask = functional_dict.get("functional_interior_missing_mask")

        @neon.Container.factory(name="IndicesBoundaryMasker_DomainBounds")
        def container_domain_bounds(
            wp_bc_indices,
            wp_id_numbers,
            wp_is_interior,
            bc_mask,
            missing_mask,
            grid_shape,
            level,
        ):
            def domain_bounds_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)

                @wp.func
                def domain_bounds_kernel(index: Any):
                    # apply the functional
                    functional_domain_bounds(
                        index,
                        wp_bc_indices,
                        wp_id_numbers,
                        wp_is_interior,
                        bc_mask_pn,
                        missing_mask_pn,
                        grid_shape,
                    )

                loader.declare_kernel(domain_bounds_kernel)

            return domain_bounds_launcher

        @neon.Container.factory(name="IndicesBoundaryMasker_InteriorBcMask")
        def container_interior_bc_mask(
            wp_bc_indices,
            wp_id_numbers,
            bc_mask,
            level,
        ):
            def interior_bc_mask_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)

                @wp.func
                def interior_bc_mask_kernel(index: Any):
                    # apply the functional
                    functional_interior_bc_mask(
                        index,
                        wp_bc_indices,
                        wp_id_numbers,
                        bc_mask_pn,
                    )

                loader.declare_kernel(interior_bc_mask_kernel)

            return interior_bc_mask_launcher

        @neon.Container.factory(name="IndicesBoundaryMasker_InteriorMissingMask")
        def container_interior_missing_mask(
            wp_bc_indices,
            bc_mask,
            missing_mask,
            grid_shape,
            level,
        ):
            def interior_bc_mask_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)

                @wp.func
                def interior_missing_mask_kernel(index: Any):
                    # apply the functional
                    functional_interior_missing_mask(
                        index,
                        wp_bc_indices,
                        bc_mask_pn,
                        missing_mask_pn,
                        grid_shape,
                    )

                loader.declare_kernel(interior_missing_mask_kernel)

            return interior_bc_mask_launcher

        container_dict = {
            "container_domain_bounds": container_domain_bounds,
            "container_interior_bc_mask": container_interior_bc_mask,
            "container_interior_missing_mask": container_interior_missing_mask,
        }

        return functional_dict, container_dict

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, bclist, bc_mask, missing_mask, start_index=None):

        grid = bc_mask.get_grid()
        num_levels = grid.num_levels
        grid_shape_finest = self.helper_masker.get_grid_shape(bc_mask)
        for level in range(num_levels):

            # Create a copy of the boundary condition list for the current level if the indices at that level are not empty
            bclist_at_level = []
            for bc in bclist:
                if bc.indices is not None and bc.indices[level]:
                    bc_copy = copy.copy(bc)  # shallow copy of the whole object
                    bc_copy.indices = copy.deepcopy(bc.indices[level])  # deep copy only the modified part
                    bclist_at_level.append(bc_copy)

            # If the boundary condition list is empty, skip to the next level
            if not bclist_at_level:
                continue

            # find grid shape at current level
            grid_shape_tuple = tuple([shape//2 ** level for shape in grid_shape_finest])
            grid_shape_warp = wp.vec3i(*grid_shape_tuple)

            # find interior boundary conditions
            bc_interior = self._find_bclist_interior(bclist_at_level, grid_shape_tuple)

            # Prepare the first kernel inputs for all items in boundary condition list
            wp_bc_indices, wp_id_numbers, wp_is_interior = self._prepare_kernel_inputs(bclist_at_level, grid_shape_tuple)

            # Launch the first container
            container_domain_bounds = self.neon_container["container_domain_bounds"](
                wp_bc_indices,
                wp_id_numbers,
                wp_is_interior,
                bc_mask,
                missing_mask,
                grid_shape_warp,
                level,
            )
            container_domain_bounds.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            # If there are no interior boundary conditions, skip the rest of the processing for this level
            if not bc_interior:
                continue

            # Prepare the second and third kernel inputs for only a subset of boundary conditions associated with the interior
            # Note 1: launching order of the following kernels are important here!
            # Note 2: Due to race conditioning, the two kernels cannot be fused together.
            wp_bc_indices, wp_id_numbers, _ = self._prepare_kernel_inputs(bc_interior, grid_shape_tuple)
            container_interior_missing_mask = self.neon_container["container_interior_missing_mask"](
                wp_bc_indices,
                bc_mask,
                missing_mask,
                grid_shape_warp,
                level,
            )
            container_interior_missing_mask.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

            # Launch the third container
            container_interior_bc_mask = self.neon_container["container_interior_bc_mask"](
                wp_bc_indices,
                wp_id_numbers,
                bc_mask,
                level,
            )
            container_interior_bc_mask.run(0, container_runtime=neon.Container.ContainerRuntime.neon)

        return bc_mask, missing_mask
