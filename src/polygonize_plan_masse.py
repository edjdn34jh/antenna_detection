import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
from shapely import affinity
import fiona                                  
import fiona.crs 

path = "/home/formation/Documents/antenna_detection/data/antenna_anfr/0592290902/059_229_0902_1_7732707.jpg"
path = "/home/formation/Documents/antenna_detection/data/antenna_anfr/0592700232/059_270_0232_1_6154523.jpg"


img = cv2.imread(path)

# Split channels
b, g, r = cv2.split(img)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s = hsv[:, :, 1]

gray_mask = s < 15


# Create output image (white background)
gray = np.ones_like(img) * 255

# Copy only original gray pixels
gray[gray_mask] = img[gray_mask]

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((50,50), np.uint8)
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


clean = clean[30:-200, 30:-30, 0]
height, width = clean.shape[:2]

contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


lengths = [cv2.arcLength(c, closed=True) for c in contours]
longest_idx = int(np.argmax(lengths))
longest_contour = contours[longest_idx]

canvas = np.ones_like(clean) * 255

cv2.drawContours(canvas,
                 longest_contour,
                 contourIdx=-1,
                 color=(0),     # blue outline (BGR)
                 thickness=5)


# Flatten the contour to a list of (x, y) tuples
pts = longest_contour.squeeze()           # shape (N, 2)
if pts.ndim != 2:                        # sometimes squeeze removes the last dim
    pts = longest_contour.reshape(-1, 2)

# Ensure the polygon is valid (Shapely will try to fix minor issues)
poly = Polygon(pts)

poly = affinity.scale(poly, xfact=1, yfact=-1, origin=(0, 0))
poly = affinity.translate(poly, xoff=0, yoff=height)

if not poly.is_valid:
    # Attempt an automatic fix – this often resolves self‑intersections
    poly = poly.buffer(0)

print(f"Polygon area (pixel²): {poly.area:.2f}")



# ------------------------------------------------------------------
# 5️⃣  Shapefile output settings
# ------------------------------------------------------------------
out_shp = "/home/formation/Documents/antenna_detection/data/contour/longest_contour.shp"

# Choose a CRS.  If you are still in image‑pixel space, EPSG:3857 or
# a custom local CRS is fine.  Replace with the correct EPSG if you
# have georeferencing info.
crs = fiona.crs.from_epsg(3857)   # Web Mercator – placeholder

schema = {
    'geometry': 'Polygon',
    'properties': {
        'id': 'int',                # simple identifier
        'perim_px': 'float',        # perimeter in pixel units
        'area_px2': 'float'         # area in pixel²
    }
}

with fiona.open(out_shp,
                mode='w',
                driver='ESRI Shapefile',
                crs=crs,
                schema=schema) as shp:
    shp.write({
        'geometry': mapping(poly),   # converts Shapely geometry → GeoJSON dict
        'properties': {
            'id': 1,
            'perim_px': float(lengths[longest_idx]),
            'area_px2': float(poly.area)
        }
    })

print(f"Shapefile written to: {out_shp}")



# cv2.imshow('image', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
