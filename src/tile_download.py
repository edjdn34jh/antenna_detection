import geopandas as gpd
import requests
import pathlib
from miscallenous import clip_tiff, rectangle_around_point

# Load the pcrs tiles data
dalles_pcrs_gdf = gpd.read_file("./dalles.geojson")


# Load the antenna data
antenna_gdf = gpd.read_file("./antenne_reproj2154.gpkg")


# Convert to same crs
dalles_pcrs_gdf = dalles_pcrs_gdf.to_crs(antenna_gdf.crs)

# 
antenna_dalles_join_gdf = gpd.sjoin(
    antenna_gdf,
    dalles_pcrs_gdf,
    how="left",
    predicate="within"
)



root_url = "https://nas-g2f.g2f-pub.smsn.fr/bloc_"
sub_dirs = ["a", "b", "c", "d", "e", "f", "gn", "gs", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r"]


all_dalle = list(set(antenna_dalles_join_gdf["nomImage"].to_list()))

dalle_name = all_dalle[1]

points = list(set((antenna_dalles_join_gdf[antenna_dalles_join_gdf["nomImage"] == dalle_name])["geometry"].tolist()))


for dir in sub_dirs:

    # antenna_dalles_join_gdf[antenna_dalles_join_gdf["nomImage"] == dalle_name]

    url = root_url + dir + '/' + dalle_name

    r = requests.get(url)

    if r.status_code == 200:
        with open(f"./pcrs_images/{dalle_name}", "wb") as f:

            f.write(r.content)

            print(f'Downloaded {url}')

for i, point in enumerate(points):

    rect = rectangle_around_point(point, 15)

    clip_tiff(
        tiff_path=f"./pcrs_images/{dalle_name}",
        geometry=rect,
        output_path=f"./antenna_images/{point}.tif"
    )

file_to_rem = pathlib.Path(f"./pcrs_images/{dalle_name}")
file_to_rem.unlink()




            


