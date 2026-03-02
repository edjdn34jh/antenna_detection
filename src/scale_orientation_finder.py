#!/usr/bin/env python3
"""
Detect orientation of an antenna ground‑plan and rotate the image so that
north points upward.

Three detection methods are available:

*   template – classic template‑matching (fast, works when the compass rose
    is clean and un‑rotated).
*   feature  – ORB/AKAZE‑based feature matching with a robust homography.
*   ocr      – detects the single capital “N” and “S” letters with Tesseract
    OCR and builds the north‑south line from them.

If the chosen method fails, the script falls back to the template method.
"""

import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# ----------------------------------------------------------------------
# Helper utilities (unchanged)
# ----------------------------------------------------------------------
def load_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    return img


def maybe_show(name: str, img: np.ndarray, base_path: Path,
               step_idx: int, show_flag: bool):
    """
    Either display the image in a window (if *show_flag* is True) or
    write it to a PNG file named
    <basename>_stepXX_<name>.png next to *base_path*.
    """
    out_path = base_path.parent / f"{base_path.stem}_step{step_idx:02d}_{name}.png"
    try:
        if show_flag:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, img)
            cv2.waitKey(0)
            cv2.destroyWindow(name)
        else:
            cv2.imwrite(str(out_path), img)
    except Exception:  # pragma: no cover – head‑less fallback
        cv2.imwrite(str(out_path), img)


def draw_rectangle(img: np.ndarray, tl: tuple[int, int], br: tuple[int, int],
                   color=(0, 0, 255)):
    """Return a colour copy of *img* with a rectangle drawn."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    cv2.rectangle(vis, tl, br, color, 2)
    return vis


# ----------------------------------------------------------------------
# OCR helper – finds the most confident N and S
# ----------------------------------------------------------------------
def ocr_find_ns(gray: np.ndarray, min_conf: int = 60):
    """
    Run Tesseract OCR on *gray* and return the centre coordinates of the
    most confident “N” and “S” characters (or None if they cannot be found).

    Parameters
    ----------
    gray : np.ndarray
        Single‑channel (uint8) image.
    min_conf : int
        Minimum OCR confidence (0‑100) to accept a character.

    Returns
    -------
    (xN, yN) or None, (xS, yS) or None
    """
    # Simple binarisation helps Tesseract on scanned drawings
    _, bin_img = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    data = pytesseract.image_to_data(
        bin_img,
        output_type=pytesseract.Output.DICT,
        config='--psm 6'          # treat image as a single block of text
    )

    best_n = best_s = None
    best_n_conf = best_s_conf = -1

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        txt = data['text'][i].strip().upper()
        conf = int(data['conf'][i])
        if conf < min_conf:
            continue
        x, y, w, h = (data['left'][i],
                       data['top'][i],
                       data['width'][i],
                       data['height'][i])
        centre = (int(x + w / 2), int(y + h / 2))

        if txt == "N" and conf > best_n_conf:
            best_n, best_n_conf = centre, conf
        elif txt == "S" and conf > best_s_conf:
            best_s, best_s_conf = centre, conf

    return best_n, best_s


# ----------------------------------------------------------------------
# Template‑matching detection (unchanged)
# ----------------------------------------------------------------------
def detect_by_template(img: np.ndarray, tmpl: np.ndarray):
    """
    Return (cx, cy, max_val, top_left, (w, h)).
    """
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    h, w = tmpl.shape
    cx = max_loc[0] + w // 2
    cy = max_loc[1] + h // 2
    return cx, cy, max_val, max_loc, (w, h)


# ----------------------------------------------------------------------
# Feature‑based detection (robust version)
# ----------------------------------------------------------------------
def detect_by_feature(img: np.ndarray, tmpl: np.ndarray,
                      *, use_clahe: bool = True,
                      detector: str = "AKAZE",
                      ratio_thresh: float = 0.75,
                      ransac_thresh: float = 8.0):
    """
    Detect the centre of *tmpl* inside *img* using a feature pipeline.
    Returns (cx, cy, inlier_ratio, H, (w, h)).
    """
    # --------------------------------------------------------------
    # 1️⃣ Optional contrast enhancement (CLAHE)
    # --------------------------------------------------------------
    def enhance_contrast(im):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(im)

    proc_img = enhance_contrast(img) if use_clahe else img.copy()
    proc_tmpl = enhance_contrast(tmpl) if use_clahe else tmpl.copy()

    # --------------------------------------------------------------
    # 2️⃣ Detector / descriptor
    # --------------------------------------------------------------
    if detector.upper() == "AKAZE":
        feat = cv2.AKAZE_create()          # binary descriptors (uint8)
    elif detector.upper() == "SIFT":
        feat = cv2.SIFT_create()           # float descriptors (float32)
    else:
        raise ValueError("Unsupported detector – choose AKAZE or SIFT")

    kp1, des1 = feat.detectAndCompute(proc_tmpl, None)
    kp2, des2 = feat.detectAndCompute(proc_img,  None)

    if des1 is None or des2 is None:
        raise RuntimeError("Feature detection failed – no descriptors produced.")

    # --------------------------------------------------------------
    # 3️⃣ FLANN matcher – choose index based on descriptor type
    # --------------------------------------------------------------
    if des1.dtype == np.uint8:                     # binary → LSH
        index_params = dict(
            algorithm = 6,          # FLANN_INDEX_LSH
            table_number = 12,
            key_size = 20,
            multi_probe_level = 2,
        )
    else:                                          # float → KD‑Tree
        index_params = dict(algorithm = 1, trees = 5)

    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # --------------------------------------------------------------
    # 4️⃣ K‑NN match + Lowe’s ratio test
    # --------------------------------------------------------------
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good) < 4:
        raise RuntimeError(
            f"Not enough good matches ({len(good)}) for homography."
        )

    # --------------------------------------------------------------
    # 5️⃣ Build point sets
    # --------------------------------------------------------------
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # --------------------------------------------------------------
    # 6️⃣ Homography with RANSAC
    # --------------------------------------------------------------
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    if H is None:
        raise RuntimeError("Homography could not be estimated.")

    # --------------------------------------------------------------
    # 7️⃣ Template centre → image centre
    # --------------------------------------------------------------
    h, w = tmpl.shape
    centre_tpl = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
    centre_img = cv2.perspectiveTransform(centre_tpl, H)[0][0]
    cx, cy = int(round(centre_img[0])), int(round(centre_img[1]))

    # --------------------------------------------------------------
    # 8️⃣ Diagnostic ratio
    # --------------------------------------------------------------
    inlier_ratio = float(mask.sum()) / mask.size

    return cx, cy, inlier_ratio, H, (w, h)


# ----------------------------------------------------------------------
# ROI extraction (unchanged)
# ----------------------------------------------------------------------
def extract_compass_region(img: np.ndarray, cx: int, cy: int, size: int = 300):
    half = size // 2
    y0, y1 = max(cy - half, 0), min(cy + half, img.shape[0])
    x0, x1 = max(cx - half, 0), min(cx + half, img.shape[1])
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


# ----------------------------------------------------------------------
# North‑arm estimation (unchanged)
# ----------------------------------------------------------------------
def estimate_north_angle(compass_roi: np.ndarray):
    edges = cv2.Canny(compass_roi, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=10,
    )
    if lines is None:
        raise RuntimeError("No lines detected in the compass region.")

    h, w = compass_roi.shape
    centre = np.array([w / 2, h / 2])

    best_line = None
    best_score = -np.inf
    for line in lines:
        x1, y1, x2, y2 = line[0]
        pt1 = np.array([x1, y1])
        pt2 = np.array([x2, y2])
        length = np.linalg.norm(pt2 - pt1)
        mid = (pt1 + pt2) / 2
        dist_to_centre = np.linalg.norm(mid - centre)
        score = length - 2 * dist_to_centre
        if score > best_score:
            best_score = score
            best_line = (pt1, pt2)

    if best_line is None:
        raise RuntimeError("Failed to pick a representative line.")

    pt1, pt2 = best_line
    vec = pt2 - pt1
    if np.dot(vec, pt2 - centre) < 0:
        vec = pt1 - pt2

    angle_rad = math.atan2(vec[0], -vec[1])   # x over -y (y grows downwards)
    angle_deg = math.degrees(angle_rad)

    # Normalise to [-180, 180]
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg < -180:
        angle_deg += 360

    return angle_deg, edges, lines, best_line


# ----------------------------------------------------------------------
# Full‑canvas rotation (now accepts a border colour)
# ----------------------------------------------------------------------
def rotate_image_full(img: np.ndarray, angle_deg: float,
                     border_color: int = 0) -> np.ndarray:
    """
    Rotate *img* by *angle_deg* (counter‑clockwise) while expanding the
    canvas so nothing gets clipped.  Returns a colour (BGR) image.
    """
    h, w = img.shape[:2]
    centre = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)

    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    M[0, 2] += (new_w / 2) - centre[0]
    M[1, 2] += (new_h / 2) - centre[1]

    rotated = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )
    return rotated


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main(
    input_path: Path,
    template_path: Path,
    method: str = "template",
    output_path: Path | None = None,
    show: bool = False,
    roi_size: int = 300,
    border_color: int = 0,
    min_ocr_conf: int = 60,
    feature_detector: str = "AKAZE",
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 8.0,
    no_clahe: bool = False,
):
    # --------------------------------------------------------------
    # Load the image (keep colour for final rotation)
    # --------------------------------------------------------------
    raw_img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    gray = (
        cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        if raw_img.ndim == 3
        else raw_img.copy()
    )

    maybe_show("0_original", gray, input_path, 0, show)

    # --------------------------------------------------------------
    # Try OCR first if requested
    # --------------------------------------------------------------
    cx = cy = None
    angle_deg = None

    if method == "ocr":
        n_pt, s_pt = ocr_find_ns(gray, min_conf=min_ocr_conf)
        if n_pt is not None and s_pt is not None:
            # Visual debugging of the two points and the line between them
            debug_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_vis, n_pt, 8, (0, 0, 255), -1)   # red = N
            cv2.circle(debug_vis, s_pt, 8, (255, 0, 0), -1)   # blue = S
            cv2.line(debug_vis, n_pt, s_pt, (0, 255, 0), 2)   # green line
            maybe_show("1_ocr_points", debug_vis, input_path, 1, show)

            # Mid‑point of N and S becomes the compass centre
            cx = int((n_pt[0] + s_pt[0]) / 2)
            cy = int((n_pt[1] + s_pt[1]) / 2)

            # Vector from S → N (north direction)
            dx = n_pt[0] - s_pt[0]
            dy = n_pt[1] - s_pt[1]

            # Same convention as the original code (clockwise from up)
            angle_rad = math.atan2(dx, -dy)   # x over -y because y grows downwards
            angle_deg = math.degrees(angle_rad)

            print(f"OCR detected N at {n_pt}, S at {s_pt}")
            print(f"Computed north‑arm angle (clockwise from up): {angle_deg:.2f}°")
        else:
            print("OCR could not reliably find BOTH N and S – falling back to template method.")
            method = "template"   # continue with the classic pipeline

    # --------------------------------------------------------------
    # Classic template / feature detection (executed if OCR failed or not selected)
    # --------------------------------------------------------------
    if method != "ocr":
        tmpl = load_gray(template_path)

        if method == "template":
            cx, cy, score, top_left, (tw, th) = detect_by_template(gray, tmpl)
            print(f"Template match score: {score:.3f}")
            match_vis = draw_rectangle(
                gray, top_left, (top_left[0] + tw, top_left[1] + th)
            )
            maybe_show("1_match_template", match_vis, input_path, 1, show)

        else:   # feature
            cx, cy, ratio, H, (tw, th) = detect_by_feature(
                gray,
                tmpl,
                use_clahe=not no_clahe,
                detector=feature_detector,
                ratio_thresh=ratio_thresh,
                ransac_thresh=ransac_thresh,
            )
            print(f"Feature‑match inlier ratio: {ratio:.2%}")
            # Visualise the projected template polygon
            corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(corners, H).astype(int)
            feat_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.polylines(
                feat_vis, [proj], isClosed=True, color=(0, 0, 255), thickness=2
            )
            maybe_show("1_match_feature", feat_vis, input_path, 1, show)

        print(f"Compass centre detected at (x={cx}, y={cy})")

        # ------------------------------------------------------------------
        # 3️⃣ Extract ROI & estimate north angle (same as before)
        # ------------------------------------------------------------------
        roi, (x0, y0, x1, y1) = extract_compass_region(gray, cx, cy, size=args.roi_size)
        roi_box_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(roi_box_vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        maybe_show("2_ROI_on_original", roi_box_vis, input_path, 2, show)

        angle_deg, edges, lines, best_line = estimate_north_angle(roi)
        print(f"Estimated north‑arm angle (clockwise from up): {angle_deg:.2f}°")

    # ------------------------------------------------------------------
    # 4️⃣ Rotate the **full** image (canvas expands)
    # ------------------------------------------------------------------
    rotation_needed = -angle_deg          # counter‑clockwise to bring north up
    rotated_full = rotate_image_full(
        raw_img,
        rotation_needed,
        border_color=args.border_color,
    )
    maybe_show("4_rotated_full_north_up", rotated_full, input_path, 4, show)

    # ------------------------------------------------------------------
    # 5️⃣ Save final result
    # ------------------------------------------------------------------
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_rotated{input_path.suffix}"
    cv2.imwrite(str(output_path), rotated_full)
    print(f"Saved rotated (full‑canvas) image to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Detect the orientation of an antenna ground‑plan and rotate the "
            "image so that north points upward.  Three detection methods are "
            "available: 'template', 'feature', and 'ocr'."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the ground‑plan image to be processed.",
    )
    parser.add_argument(
        "--template",
        required=True,
        type=Path,
        help="Path to the compass‑rose template image (used for template/feature methods).",
    )
    parser.add_argument(
        "--method",
        choices=["template", "feature", "ocr"],
        default="template",
        help="Detection technique (default: template).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the rotated image (default: <input>_rotated.<ext>).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display intermediate images; if the environment is head‑less the images are saved as PNG files.",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=300,
        help="Side length of the square ROI extracted around the detected compass centre (pixels).",
    )
    parser.add_argument(
        "--border-color",
        type=int,
        default=0,
        help="Pixel value used for the padding border when rotating (0 = black).",
    )
    parser.add_argument(
        "--min-ocr-conf",
        type=int,
        default=60,
        help="Minimum Tesseract confidence (0‑100) required for an 'N' or 'S' to be accepted.",
    )
    parser.add_argument(
        "--feature-detector",
        choices=["AKAZE", "SIFT"],
        default="AKAZE",
        help="Descriptor algorithm for the feature method.",
    )
    parser.add_argument(
        "--ratio-thresh",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold for feature matching.",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=8.0,
        help="RANSAC reprojection error tolerance (pixels) for homography estimation.",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE contrast enhancement for the feature method.",
    )

    args = parser.parse_args()

    # Configure a simple console logger (INFO level)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    main(
        input_path=args.input,
        template_path=args.template,
        method=args.method,
        output_path=args.output,
        show=args.show,
        roi_size=args.roi_size,
        border_color=args.border_color,
        min_ocr_conf=args.min_ocr_conf,
        feature_detector=args.feature_detector,
        ratio_thresh=args.ratio_thresh,
        ransac_thresh=args.ransac_thresh,
        no_clahe=args.no_clahe,
    )