# For odd lines, calculate all shifted positions at once with interpolation
if y + 1 < height:
    indices = xp.mod(xp.arange(width) + phase_map / 360.0 * width, width)

    # For numpy implementation (CPU)
    if xp is np:
        from scipy.interpolate import interp1d

        interpolator = interp1d(
            np.arange(width),
            image[f, y + 1, :],
            bounds_error=False,
            fill_value="extrapolate",
        )
        result[f, y + 1, :] = interpolator(indices)
    # For CuPy implementation (GPU)
    else:
        # Use CuPy's built-in interpolation
        result[f, y + 1, :] = cp.interp(indices, cp.arange(width), image[f, y + 1, :])