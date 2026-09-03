"""
Mono calibration of the RGB camera -- uses the same images captured for lidar-camera extrinsics.
Writes results/rgb_intrinsics.yaml in the format expected by common.load_mono_yaml (K, dist, image_size).
"""
import glob, os
import cv2
import numpy as np

from common import find_chessboard

COLS_INNER, ROWS_INNER = 9, 6
SQUARE_SIZE_MM = 29.0
RGB_DIR = "calib_images/rgb"
OUTPUT_PATH = "results/rgb_intrinsics.yaml"

def build_object_points():
    objp = np.zeros((ROWS_INNER * COLS_INNER, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:COLS_INNER, 0:ROWS_INNER].T.reshape(-1, 2)
    objp *= (SQUARE_SIZE_MM / 1000.0)  # meters, consistent with the lidar side
    return objp

def calibrate():
    objp = build_object_points()
    obj_points, img_points = [], []
    image_size = None

    rgb_files = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")))
    for rgb_path in rgb_files:
        img = cv2.imread(rgb_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        found, corners = find_chessboard(gray, COLS_INNER, ROWS_INNER)
        if not found:
            continue
        obj_points.append(objp)
        img_points.append(corners)

    print(f"Used {len(obj_points)} / {len(rgb_files)} images for calibration")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    print(f"RMS reprojection error: {rms:.4f}")
    print("K:\n", K)
    print("dist:\n", dist)

    os.makedirs("results", exist_ok=True)
    fs = cv2.FileStorage(OUTPUT_PATH, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("dist", dist)
    fs.write("image_size", np.array(image_size, dtype=np.int32))
    fs.release()
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    calibrate()