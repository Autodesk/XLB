import warp as wp

class ColorMapper:
    """
    Operator for mapping scalar values to RGB colors using different colormaps.
    
    Currently supported colormaps:
    - 'jet': Blue -> Cyan -> Yellow -> Red
    """
    
    @staticmethod
    @wp.func
    def jet_colormap(value: float) -> wp.vec3:
        """
        Map a value in [0,1] to RGB colors using the jet colormap.
        """
        r = wp.clamp(wp.min(4.0 * value - 1.5, -4.0 * value + 4.5), 0.0, 1.0)
        g = wp.clamp(wp.min(4.0 * value - 0.5, -4.0 * value + 3.5), 0.0, 1.0)
        b = wp.clamp(wp.min(4.0 * value + 0.5, -4.0 * value + 2.5), 0.0, 1.0)
        
        return wp.vec3(r, g, b)
    
    @wp.kernel
    def _map_colors(
        values: wp.array(dtype=float),
        colors: wp.array2d(dtype=float),
        vmin: float,
        vmax: float,
    ):
        # Get thread index
        idx = wp.tid()
        
        # Normalize value to [0,1] range
        value = values[idx]
        normalized = (value - vmin) / (vmax - vmin)
        normalized = wp.clamp(normalized, 0.0, 1.0)
        
        # Map to color using jet colormap
        color = ColorMapper.jet_colormap(normalized)
        
        # Store RGB values
        colors[idx, 0] = color[0]
        colors[idx, 1] = color[1]
        colors[idx, 2] = color[2]
    
    def __call__(
        self,
        values: wp.array,
        colors: wp.array2d,
        vmin: float,
        vmax: float,
        colormap: str = 'jet',
    ) -> wp.array2d:
        """
        Map scalar values to RGB colors using the specified colormap.
        
        Parameters
        ----------
        values : wp.array(dtype=float)
            Input scalar values to map to colors
        colors : wp.array2d(dtype=float)
            Output RGB colors array with shape (len(values), 3)
        vmin : float
            Minimum value for normalization
        vmax : float
            Maximum value for normalization
        colormap : str
            Name of the colormap to use (currently only 'jet' is supported)
            
        Returns
        -------
        wp.array2d(dtype=float)
            Reference to the input colors array
        """
        if colormap != 'jet':
            raise ValueError(f"Colormap '{colormap}' not supported. Use 'jet'.")
            
        # Verify input shapes
        assert len(values.shape) == 1, "Values must be a 1D array"
        assert colors.shape == (len(values), 3), f"Colors array must have shape ({len(values)}, 3)"
        
        # Verify vmin/vmax
        assert isinstance(vmin, float), "vmin must be a float"
        assert isinstance(vmax, float), "vmax must be a float"
        assert vmax > vmin, f"vmax ({vmax}) must be greater than vmin ({vmin})"
        
        # Launch kernel
        wp.launch(
            self._map_colors,
            dim=len(values),
            inputs=[values, colors, vmin, vmax],
        )
        
        return colors 