"""
Point-to-plane LiDAR-camera extrinzicna kalibracija -- CISTA verzija,
bez izmisljenih ogranicenja i bez rucnih offset hakova.

Kamera strana: solvePnP na chessboard uglovima daje normalu i tacku
ravni table U KAMERA koordinatama.
Lidar strana: PCA fit ravni nad ROI tackama daje SAMO inlier tacke
ravni (bez uglova/redosleda/interpolacije mreze).

Cost funkcija: point-to-plane rastojanje, SVI frejmovi odjednom.
Dva prolaza: prvi nadje grubu ocenu i odbaci outlier frejmove (RMS
prevelik), drugi re-optimizuje samo nad dobrim frejmovima.

Pokretanje:
    python3 point_plane_calibration.py
"""
import glob, os, re
import cv2
import numpy as np
from scipy.optimize import least_squares

COLS_INNER, ROWS_INNER = 9, 6
SQUARE_SIZE_M = 29.0 / 1000.0
RGB_DIR = "calib_images/rgb"
LIDAR_DIR = "calib_images/lidar"
RGB_INTRINSICS = "results/rgb_intrinsics.yaml"
OUTPUT_PATH = "results/lidar_camera_extrinsics.yaml"

ROI_CENTER = np.array([1.0, 0.0, 0.0])
ROI_HALF_SIZE = 0.6
PLANE_DIST_THRESHOLD = 0.008
OUTLIER_RMS_THRESHOLD_MM = 12.0
N_RANDOM_RESTARTS = 6


def load_mono_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    fs.release()
    return K, dist


def build_object_points():
    objp = np.zeros((ROWS_INNER * COLS_INNER, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:COLS_INNER, 0:ROWS_INNER].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M
    return objp


def load_pcd_ascii(path):
    data = np.loadtxt(path, skiprows=11, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    points = data[:, :3]
    intensity = data[:, 3] if data.shape[1] > 3 else np.zeros(len(data), dtype=np.float32)
    mask = np.isfinite(points).all(axis=1) & np.isfinite(intensity)
    return points[mask], intensity[mask]


def find_chessboard(gray_img):
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
              + cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray_img, (COLS_INNER, ROWS_INNER), flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
    return found, corners


BOARD_CENTER_OBJ = np.array([(COLS_INNER - 1) * SQUARE_SIZE_M / 2.0,
                              (ROWS_INNER - 1) * SQUARE_SIZE_M / 2.0, 0.0])

def board_plane_in_camera(img_pts, obj_pts, K, dist):
    """Normala, tacka ravni i centar table, U KAMERA koordinatama."""
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    normal_cam = R[:, 2]
    point_cam = tvec.flatten()
    center_cam = R @ BOARD_CENTER_OBJ + point_cam
    return normal_cam, point_cam, center_cam


# Fizicke dimenzije table (9x6 unutrasnji grid + margine, ~10x7 kvadrata
# od 29mm). Blago uvecano za toleranciju.
BOARD_WIDTH_M = 10 * 0.029 * 1.15
BOARD_HEIGHT_M = 7 * 0.029 * 1.15


def extract_lidar_plane_points(pcd_path):
    """SAMO inlier tacke ravni table (lidar frejm) -- PCA fit + filter
    preko ORIJENTISANOG PRAVOUGAONIKA (ne kruga) velicine table.
    Pod/zid koji dodiruju ivicu table se sire u JEDNOM pravcu (uzan
    'rep'), pa krug oko centra to lako propusti cak i sa malim
    radijusom -- pravougaonik poravnat sa glavnim osama table je mnogo
    strozi jer prati stvarni oblik i orijentaciju table u ravni."""
    points, intensity = load_pcd_ascii(pcd_path)
    mask = np.all(np.abs(points - ROI_CENTER) < ROI_HALF_SIZE, axis=1)
    roi_points = points[mask]
    roi_intensity = intensity[mask]
    if len(roi_points) < 30:
        return None

    if roi_intensity.max() > roi_intensity.min():
        cutoff = np.percentile(roi_intensity, 70.0)
        bright_mask = roi_intensity >= cutoff
        candidate = roi_points[bright_mask] if bright_mask.sum() >= 30 else roi_points
    else:
        candidate = roi_points

    centroid = candidate.mean(axis=0)
    centered = candidate - centroid
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, np.argmin(evals)]
    normal /= np.linalg.norm(normal)

    dists = np.abs(centered @ normal)
    coplanar = dists < PLANE_DIST_THRESHOLD
    coplanar_pts = candidate[coplanar]
    if len(coplanar_pts) < 10:
        return None

    # 2D projekcija koplanarnih tacaka u ravan (u,v bazis oko normale)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(normal, arbitrary); u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    board_centroid = coplanar_pts.mean(axis=0)
    rel = coplanar_pts - board_centroid
    uv = np.stack([rel @ u_axis, rel @ v_axis], axis=1).astype(np.float32)

    # PCA u samoj uv ravni da se nadje glavna orijentacija table (jer
    # rectangle nije nuzno poravnat sa u_axis/v_axis izborom iznad).
    uv_cov = np.cov(uv.T)
    uv_evals, uv_evecs = np.linalg.eigh(uv_cov)
    major_axis = uv_evecs[:, np.argmax(uv_evals)]
    minor_axis = np.array([-major_axis[1], major_axis[0]])
    uv_center = uv.mean(axis=0)
    rel_uv = uv - uv_center
    along_major = rel_uv @ major_axis
    along_minor = rel_uv @ minor_axis

    # Pravougaonik je poluopseg u svakom pravcu -- veca dimenzija table
    # ide uz major_axis (deteknovana najveca varijansa), manja uz minor.
    half_major = max(BOARD_WIDTH_M, BOARD_HEIGHT_M) / 2
    half_minor = min(BOARD_WIDTH_M, BOARD_HEIGHT_M) / 2
    on_board_2d = (np.abs(along_major) < half_major) & (np.abs(along_minor) < half_minor)

    inlier_pts = coplanar_pts[on_board_2d]
    if len(inlier_pts) < 20:
        return None
    lidar_center = inlier_pts.mean(axis=0)
    return inlier_pts.astype(np.float64), lidar_center.astype(np.float64)


def collect_frames():
    K, dist = load_mono_yaml(RGB_INTRINSICS)
    obj_pts = build_object_points()
    rgb_files = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")))

    frames = []
    for rgb_path in rgb_files:
        basename = os.path.splitext(os.path.basename(rgb_path))[0]
        idx_match = re.search(r'(\d+)$', basename)
        if not idx_match:
            continue
        idx = idx_match.group(1)
        pcd_path = os.path.join(LIDAR_DIR, f"lidar_{idx}.pcd")
        if not os.path.exists(pcd_path):
            continue

        gray = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2GRAY)
        found, corners = find_chessboard(gray)
        if not found:
            continue
        img_pts = corners.reshape(-1, 2).astype(np.float64)

        cam_plane = board_plane_in_camera(img_pts, obj_pts, K, dist)
        if cam_plane is None:
            continue
        normal_cam, point_cam, center_cam = cam_plane

        lidar_result = extract_lidar_plane_points(pcd_path)
        if lidar_result is None:
            print(f"  [!] {basename}: nedovoljno lidar plane tacaka, preskacem")
            continue
        lidar_pts, lidar_center = lidar_result

        frames.append((basename, normal_cam, point_cam, lidar_pts, center_cam, lidar_center))

    return frames, K, dist


CENTER_WEIGHT = 40.0  # balansira centar-constraint protiv mnostva plane tacaka

def residuals(params, frames):
    """Point-to-plane + centar-table point-to-point (resava neodredjenost
    translacije unutar ravni koju plane-only cost ne vidi)."""
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:6].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()

    all_res = []
    for name, normal_cam, point_cam, lidar_pts, center_cam, lidar_center in frames:
        transformed = (R @ lidar_pts.T).T + t
        plane_res = (transformed - point_cam) @ normal_cam
        all_res.append(plane_res)

        transformed_center = R @ lidar_center + t
        center_res = CENTER_WEIGHT * (transformed_center - center_cam)
        all_res.append(center_res)
    return np.concatenate(all_res)


def per_frame_rms(params, frames):
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:6].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)

    results = []
    for name, normal_cam, point_cam, lidar_pts, center_cam, lidar_center in frames:
        transformed = (R @ lidar_pts.T).T + tvec.flatten()
        res = (transformed - point_cam) @ normal_cam
        rms = np.sqrt(np.mean(res ** 2))
        results.append((name, rms))
    return results

def optimize(frames):
    rng = np.random.default_rng(42)
    x0_list = [np.zeros(6)]
    for _ in range(N_RANDOM_RESTARTS):
        rvec0 = rng.uniform(-np.pi, np.pi, size=3)
        tvec0 = rng.uniform(-0.3, 0.3, size=3)
        x0_list.append(np.concatenate([rvec0, tvec0]))

    best_result, best_cost = None, None
    for x0 in x0_list:
        result = least_squares(residuals, x0, args=(frames,),
                                method="trf", loss="soft_l1", f_scale=0.05)
        if best_cost is None or result.cost < best_cost:
            best_cost, best_result = result.cost, result
    return best_result


def main():
    print("Sakupljanje uparenih podataka...")
    frames, K, dist = collect_frames()
    print(f"Validnih frejmova u pocetku: {len(frames)}")
    if len(frames) < 4:
        print("Nedovoljno frejmova za stabilnu kalibraciju.")
        return

    print("\n--- PROLAZ 1: Inicijalna optimizacija ---")
    result1 = optimize(frames)
    rms_list = per_frame_rms(result1.x, frames)

    good_frames = [f for f, (name, rms) in zip(frames, rms_list)
                    if rms * 1000 <= OUTLIER_RMS_THRESHOLD_MM]
    bad = [(name, rms) for name, rms in rms_list if rms * 1000 > OUTLIER_RMS_THRESHOLD_MM]
    if bad:
        print(f"Izbaceno {len(bad)} outlier frejmova sa RMS > {OUTLIER_RMS_THRESHOLD_MM}mm:")
        for name, rms in bad:
            print(f"  [X] {name}: RMS={rms*1000:.2f}mm")

    if len(good_frames) < 4:
        print("Premalo dobrih frejmova posle filtriranja, koristim sve iz prolaza 1.")
        good_frames = frames
    elif len(good_frames) < 10:
        print(f"UPOZORENJE: samo {len(good_frames)} frejmova ispod praga od "
              f"{OUTLIER_RMS_THRESHOLD_MM}mm -- moze biti nedovoljno geometrijske "
              f"raznovrsnosti za pouzdan fit (razmisli o blagom podizanju praga).")

    normals = np.array([f[1] for f in good_frames])
    cos_angles = normals @ normals.T
    np.fill_diagonal(cos_angles, 1.0)
    min_cos = cos_angles.min()
    print(f"\nMax ugao izmedju board normala u dobrim frejmovima: "
          f"{np.degrees(np.arccos(np.clip(min_cos, -1, 1))):.1f} deg")
    if np.degrees(np.arccos(np.clip(min_cos, -1, 1))) < 15:
        print("UPOZORENJE: normale su skoro paralelne -- translacija unutar "
              "ravni table je slabo ogranicena, rezultat je nepouzdan.")

    print(f"\n--- PROLAZ 2: Re-optimizacija na {len(good_frames)} dobrih frejmova ---")
    result2 = optimize(good_frames)

    rvec = result2.x[:3].reshape(3, 1)
    tvec = result2.x[3:6].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)

    print("\n--- Per-frame point-to-plane RMS (posle filtriranja) ---")
    for name, rms in per_frame_rms(result2.x, good_frames):
        print(f"  {name}: RMS={rms*1000:.2f}mm")

    overall_rms = np.sqrt(np.mean(residuals(result2.x, good_frames) ** 2))
    print(f"\nGlobalni RMS (ocisceni frejmovi): {overall_rms*1000:.2f}mm")
    print("Rotation (lidar->camera):\n", R)
    print("Translation (lidar->camera):\n", tvec)

    os.makedirs("results", exist_ok=True)
    fs = cv2.FileStorage(OUTPUT_PATH, cv2.FILE_STORAGE_WRITE)
    fs.write("R", R)
    fs.write("rvec", rvec)
    fs.write("tvec", tvec)
    fs.release()
    print(f"\nSacuvano u {OUTPUT_PATH}")


if __name__ == "__main__":
    main()