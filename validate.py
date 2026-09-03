import cv2
import numpy as np

from common import load_mono_yaml
from extract_lidar_board import load_pcd

ROI_CENTER = np.array([1.0, 0.0, 0.0])
ROI_HALF_SIZE = 0.6
CHESSBOARD_WIDTH = 10 * 0.029
CHESSBOARD_HEIGHT = 7 * 0.029
BOARD_PLANE_TOLERANCE = 0.025


def fit_plane_from_points(points, dist_threshold=0.015):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 10:
        return None, None

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    evals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(evals)]
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal

    d = -normal.dot(centroid)
    distances = np.abs(points @ normal + d)
    inliers = np.where(distances <= dist_threshold)[0]
    return normal, inliers


def board_polygon_2d_from_roi(roi_points):
    """Return the known chessboard rectangle in fitted-plane coordinates."""
    normal, inliers = fit_plane_from_points(roi_points, dist_threshold=0.015)
    if normal is None or len(inliers) < 20:
        return None, None, None, None

    inlier_pts = roi_points[inliers]
    centroid = inlier_pts.mean(axis=0)
    if np.dot(normal, -centroid) < 0:
        normal = -normal

    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(normal, arbitrary)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)

    rel = inlier_pts - centroid
    uv = np.stack([rel @ u_axis, rel @ v_axis], axis=1).astype(np.float32)

    covariance = np.cov(uv.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    width_axis = eigenvectors[:, np.argmax(eigenvalues)]
    height_axis = np.array([-width_axis[1], width_axis[0]])
    center_uv = uv.mean(axis=0)
    half_width = width_axis * (CHESSBOARD_WIDTH / 2)
    half_height = height_axis * (CHESSBOARD_HEIGHT / 2)
    box = np.array([
        center_uv - half_width - half_height,
        center_uv + half_width - half_height,
        center_uv + half_width + half_height,
        center_uv - half_width + half_height,
    ], dtype=np.float32)

    return centroid, normal, u_axis, v_axis, box


def validate(rgb_path, pcd_path, extrinsics_path="results/lidar_camera_extrinsics.yaml"):
    K, dist, _ = load_mono_yaml("results/rgb_intrinsics.yaml")

    fs = cv2.FileStorage(extrinsics_path, cv2.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    fs.release()

    data = np.loadtxt(pcd_path, skiprows=11, dtype=np.float32)
    points = data[:, :3]
    intensity = data[:, 3]
    finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(intensity)
    points = points[finite_mask]
    intensity = intensity[finite_mask]

    roi_mask = np.all(np.abs(points - ROI_CENTER) < ROI_HALF_SIZE, axis=1)
    roi_points = points[roi_mask]
    roi_intensity = intensity[roi_mask]
    if len(roi_points) < 20:
        raise ValueError(f"Not enough points in ROI for {pcd_path}")

    intensity_cutoff = np.percentile(roi_intensity, 70.0)
    white_mask = roi_intensity >= intensity_cutoff
    white_points = roi_points[white_mask]
    plane_points = white_points if len(white_points) >= 20 else roi_points

    centroid, normal, u_axis, v_axis, board_box = board_polygon_2d_from_roi(plane_points)
    if centroid is None:
        raise ValueError(f"Could not fit board rectangle for {pcd_path}")

    rel = roi_points - centroid
    uv_all = np.stack([rel @ u_axis, rel @ v_axis], axis=1)
    plane_distance = np.abs(rel @ normal)
    board_mask = []
    for p, distance in zip(uv_all, plane_distance):
        inside = cv2.pointPolygonTest(board_box, (float(p[0]), float(p[1])), False) >= 0
        board_mask.append(inside and distance <= BOARD_PLANE_TOLERANCE)
    board_mask = np.array(board_mask, dtype=bool)

    board_points = roi_points[board_mask].astype(np.float32)
    if len(board_points) < 20:
        raise ValueError(f"Board footprint too small after polygon filtering for {pcd_path}")
    print(f"Projecting {len(board_points)} LiDAR board points")

    img_points, _ = cv2.projectPoints(board_points, rvec, tvec, K, dist)
    img_points = img_points.reshape(-1, 2)

    img = cv2.imread(rgb_path)
    h, w = img.shape[:2]
    for pt in img_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Validation: only board footprint projected onto RGB", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    import sys
    validate(sys.argv[1], sys.argv[2])