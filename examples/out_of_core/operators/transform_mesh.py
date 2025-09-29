import warp as wp

class TransformMesh:
    """
    Operator for transforming mesh vertices using translation and scaling.
    
    The transformation is applied in the following order:
    1. Scale around origin
    2. Translate to new origin
    """
    
    @wp.kernel
    def _transform_vertices(
        vertices: wp.array(dtype=wp.vec3),
        origin: wp.vec3,
        scale: wp.vec3,
    ):
        # Get thread index
        idx = wp.tid()
        
        # Get vertex
        vertex = vertices[idx]
        
        # Apply scale
        vertex = wp.vec3(
            vertex[0] * scale[0],
            vertex[1] * scale[1],
            vertex[2] * scale[2]
        )
        
        # Apply translation
        vertex = vertex + origin
        
        # Store result
        vertices[idx] = vertex

    def __call__(
        self,
        mesh: wp.Mesh,
        origin: wp.vec3,
        scale: wp.vec3,
    ) -> wp.Mesh:
        """
        Transform mesh vertices using translation and scaling.
        
        Parameters
        ----------
        mesh : wp.Mesh
            Input mesh to transform
        origin : wp.vec3
            New origin for the mesh (translation)
        scale : wp.vec3
            Scale factors for each axis
            
        Returns
        -------
        wp.Mesh
            New mesh with transformed vertices
        """
        # Create new vertices array
        new_vertices = wp.clone(mesh.points)
        
        # Launch kernel to transform vertices
        wp.launch(
            self._transform_vertices,
            dim=new_vertices.shape[0],
            inputs=[new_vertices, origin, scale],
        )
        
        # Create new mesh with transformed vertices
        transformed_mesh = wp.Mesh(
            points=new_vertices,
            indices=mesh.indices,
        )
        
        return transformed_mesh 