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
