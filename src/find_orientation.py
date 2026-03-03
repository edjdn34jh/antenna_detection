import cv2
import numpy as np


def extract_scale_rotation(H):
    """
    Extract approximate scale and rotation from homography matrix.
    Assumes weak perspective distortion.
    """

    # Normalize so that H[2,2] = 1
    H = H / H[2,2]

    # Extract affine part
    a = H[0,0]
    b = H[0,1]
    c = H[1,0]
    d = H[1,1]

    # Compute scale (average of x and y scale)
    scale_x = np.sqrt(a*a + c*c)
    scale_y = np.sqrt(b*b + d*d)
    scale = (scale_x + scale_y) / 2

    # Compute rotation (radians)
    theta = np.arctan2(c, a)

    # Convert to degrees
    rotation_deg = np.degrees(theta)

    return scale, rotation_deg


def poi_matching(img_path, template_path, display_results=False)
    # Load the main image and the template image
    img = cv2.imread('/host/img_antenna/data/antenna_tiles/antenna_anfr_masse/0592291161/059_229_1161_1_7320205_PLAN_DE_MASSE.jpg', cv2.IMREAD_GRAYSCALE)
    template = cv2.imread('/host/img_antenna/data/compasse_template_1.png', cv2.IMREAD_GRAYSCALE)


    # -----------------------------
    # 1. Load images
    # -----------------------------
    img1 = template
    img2 = img

    if img1 is None or img2 is None:
        raise ValueError("Error loading images!")

    # -----------------------------
    # 2. Initialize SIFT detector
    # -----------------------------
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(f"Keypoints in template: {len(kp1)}")
    print(f"Keypoints in scene: {len(kp2)}")

    # -----------------------------
    # 3. FLANN-based matcher
    # -----------------------------
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # -----------------------------
    # 4. Lowe's Ratio Test
    # -----------------------------
    good_matches = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Good matches after ratio test: {len(good_matches)}")

    # -----------------------------
    # 5. RANSAC + Homography
    # -----------------------------
    MIN_MATCH_COUNT = 10

    if len(good_matches) > MIN_MATCH_COUNT:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        matches_mask = mask.ravel().tolist()

        # Get template dimensions
        h, w = img1.shape

        # Define template corners
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

        # Transform corners to scene
        dst = cv2.perspectiveTransform(pts, H)

        # Draw detected object in scene
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        img2_detected = cv2.polylines(img2_color, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)

        print("Homography Matrix:\n", H)


        scale, rotation = extract_scale_rotation(H)

        print(f"Estimated Scale: {scale}")
        print(f"Estimated Rotation (degrees): {rotation}")


    else:
        print("Not enough matches found!")
        matches_mask = None
        img2_detected = img2
        scale, rotation, H = None, None, None


    # -----------------------------
    # 6. Draw matches
    # -----------------------------
    if display_results:

        draw_params = dict(
            matchColor=(0,255,0),
            singlePointColor=None,
            matchesMask=matches_mask,
            flags=2
        )

        img_matches = cv2.drawMatches(
            img1, kp1,
            img2_detected, kp2,
            good_matches, None,
            **draw_params
        )

        # -----------------------------
        # 7. Show Results
        # -----------------------------
        cv2.imshow("Matches with RANSAC", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return(scale, rotation, H)
