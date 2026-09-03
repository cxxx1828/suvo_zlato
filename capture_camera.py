"""
RealSense RGB capture for lidar-camera calibration.
Requires: pip install pyrealsense2 opencv-python numpy
"""
import os, glob, cv2, numpy as np
import pyrealsense2 as rs

from common import find_chessboard

COLS_INNER, ROWS_INNER = 9, 6
RGB_DIR = "calib_images/rgb"
LIDAR_TRIGGER_FILE = "calib_images/lidar/.trigger"

def capture():
    os.makedirs(RGB_DIR, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    pipeline.start(config)

    count = len(glob.glob(os.path.join(RGB_DIR, "*.png")))
    print("SPACE = save frame (remember the timestamp for the lidar!), Q = quit")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError:
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb_bgr = np.asanyarray(color_frame.get_data())

            found, corners = find_chessboard(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY), COLS_INNER, ROWS_INNER)
            disp = rgb_bgr.copy()
            if found:
                cv2.drawChessboardCorners(disp, (COLS_INNER, ROWS_INNER), corners, found)
            cv2.putText(disp, f"frame {count:03d} board:{'OK' if found else 'NO'}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow("RealSense capture", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and found:
                basename = f"img_{count:03d}.png"
                cv2.imwrite(os.path.join(RGB_DIR, basename), rgb_bgr)

                # Trigger lidar_capture_tool to save the CORRESPONDING .pcd --
                # atomic write (tmp file + rename) so the lidar side never
                # reads a partially written trigger.
                tmp_path = LIDAR_TRIGGER_FILE + ".tmp"
                with open(tmp_path, "w") as f:
                    f.write(f"{count:03d}")
                os.replace(tmp_path, LIDAR_TRIGGER_FILE)

                print(f"Saved {basename} -- lidar capture triggered with the same number")
                count += 1

            elif key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture()