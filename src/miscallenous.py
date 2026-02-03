######################################################################################################
########################## Some miscallenous function used in the  ###################################
##########################          tile_download script           ###################################
######################################################################################################



import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd



def clip_tiff(tiff_path, geometry, output_path):
    """
    Clip a tiff according to a geometry
    
    :param tiff_path: The path to the input tiff
    :param geometry: The geometry to clip the tiff
    :param output_path: The path to save the clipped tiff
    """
    with rasterio.open(tiff_path) as src:

        out_image, out_transform = mask(
            src,
            [geometry],
            crop=True
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)




def rectangle_around_point(point, half_size):
    """
    Given a point and dimension, create a square geometry centered around that point
    
    :param point: The point around which to center the geometry
    :param half_size: The halfe size of the square
    """

    x, y = point.x, point.y
    return box(
        x - half_size,
        y - half_size,
        x + half_size,
        y + half_size
    )



class TilePathError(TypeError):
    """Raised when a component of a tile‑path is not a string."""
    def __init__(self, component_name: str, bad_value):
        msg = (
            f"Unable to build the tile URL because the value for "
            f"`{component_name}` is not a string (got {type(bad_value)!r}).\n"
            "Typical causes:\n"
            "- A numeric column (e.g. a float) was read from a CSV/DB.\n"
            "- An earlier calculation produced a NaN/None that got cast to float.\n"
            "\nFix the upstream data or cast the value to `str` before concatenation."
        )
        super().__init__(msg)


def safe_concat(*parts: tuple) -> str:
    """
    Concatenate path components safely.

    Parameters
    ----------
    *parts : tuple
        Any number of values that should form a URL/path.  All parts must be
        convertible to ``str`` *without* being a ``float`` (including ``nan``).

    Returns
    -------
    str
        The concatenated string.

    Raises
    ------
    TilePathError
        If any part is a ``float`` (or ``np.nan``) – the most common source of the
        “can only concatenate str (not "float") to str” error.
    """
    cleaned_parts = []
    for i, p in enumerate(parts):
        # Detect floats (including numpy.float64, pandas Float64, math.nan, etc.)
        if isinstance(p, float):
            raise TilePathError(f"part #{i+1}", p)

        # Anything else we coerce to string – this also handles None gracefully.
        cleaned_parts.append("" if p is None else str(p))

    return "".join(cleaned_parts)