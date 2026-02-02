######################################################################################################
########################## Python script to automatically download pcrs .tif #########################
##########################    tiles and clip them around selected antenna    #########################
######################################################################################################


import geopandas as gpd
import requests
import pathlib
from miscallenous import clip_tiff, rectangle_around_point

# Load the pcrs tiles data
pcrs_tiles_gdf = gpd.read_file("./data/dalles.geojson")


# Load the antenna data
antenna_gdf = gpd.read_file("./data/hauts_de_france.gpkg")


# Convert to same crs
pcrs_tiles_gdf = pcrs_tiles_gdf.to_crs(antenna_gdf.crs)

# Join the antenna and tile dataframe
# The resulting dataframe is composed of all the antennas and for each, the tile that it seats on
antenna_tiles_join_gdf = gpd.sjoin(
    antenna_gdf,
    pcrs_tiles_gdf,
    how="left",
    predicate="within"
)


# The url where the pcrs tiles are stored
root_url = "https://nas-g2f.g2f-pub.smsn.fr/bloc_"
sub_dirs = ["a", "b", "c", "d", "e", "f", "gn", "gs", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r"]


# List of all uniques tiles where there are antennas
all_tiles = list(set(antenna_tiles_join_gdf["nomImage"].to_list()))


skip_loop = True

# Iterates over all individual tiles containing at least an antenna
for tile_name in all_tiles:

    downloaded = False

    if tile_name == "HFC-2022-0766-6987-LA93-0M05-RVB.tif":
        
        skip_loop = False
        print("Starting dowload from {tile_name}")

    if skip_loop:
        next

    else:
        # List of all unique antennas that seats within the given tile
        points = antenna_tiles_join_gdf.loc[antenna_tiles_join_gdf["nomImage"] == tile_name,["geometry", "sta_nm_anfr"]]
        points = set([
            (geom, sta)          # geom is a Shapely geometry object
            for geom, sta in zip(points["geometry"], points["sta_nm_anfr"])
        ])



        # Download the chosen tile
        for dir in sub_dirs:

            url = root_url + dir + '/' + tile_name

            r = requests.get(url)

            if r.status_code == 200:
                with open(f"/host/img_antenna/data/pcrs_tiles_tmp/{tile_name}", "wb") as f:

                    f.write(r.content)

                    print(f'Downloaded {url}')

                    downloaded = True


        # Clip a 60x60m square around the points in the tiles and save them as tif files
        if downloaded:
            for i, point in enumerate(points):

                rect = rectangle_around_point(point[0], 30)

                clip_tiff(
                    tiff_path=f"/host/img_antenna/data/pcrs_tiles_tmp/{tile_name}",
                    geometry=rect,
                    output_path=f"/host/img_antenna/data/antenna_tiles/{point[1]}.tif"
                )

            # Remove the dowloaded tile to save memory space
            file_to_rem = pathlib.Path(f"/host/img_antenna/data/pcrs_tiles_tmp/{tile_name}")
            file_to_rem.unlink()




            


