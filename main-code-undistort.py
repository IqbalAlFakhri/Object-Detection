import cv2
import numpy as np
import torch
from ultralytics import YOLOv10
from triangulasi import calculate_distance
from readKalibrasi import undistortRectify  # Import fungsi undistortRectify

# Cek apakah GPU tersedia, jika tidak gunakan CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model YOLOv10
model_path = "model-1.pt"
model = YOLOv10(model_path).to(device)

# Buka kamera:
# Kamera kanan (deteksi + segmentasi) menggunakan index 0
# Kamera kiri (deteksi saja) menggunakan index 2
cap_right = cv2.VideoCapture(2)
cap_left = cv2.VideoCapture(0)

# Atur ukuran frame masing-masing kamera menjadi 640x480
frame_width, frame_height = 640, 480
for cap in [cap_right, cap_left]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()
    if not ret_right or not ret_left:
        break

    # Resize frame jika diperlukan
    frame_right = cv2.resize(frame_right, (frame_width, frame_height))
    frame_left = cv2.resize(frame_left, (frame_width, frame_height))

    # === UNDISTORT DAN REKTIFIKASI ===
    # Memperbaiki distorsi gambar dengan menggunakan file kalibrasi (XML)
    frame_right, frame_left = undistortRectify(frame_right, frame_left)

    # === PROSES KAMERA KIRI (Deteksi Saja) ===
    detection_left = frame_left.copy()
    left_detections = []  # List untuk menyimpan koordinat deteksi dari kamera kiri

    results_left = model(frame_left, device=device)
    for result in results_left:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for idx, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = map(int, box[:4])
            center_x_left = int((x1 + x2) / 2)
            center_y_left = int((y1 + y2) / 2)
            left_detections.append({'center_x': center_x_left, 'center_y': center_y_left})
            cv2.rectangle(detection_left, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(detection_left, f"{conf*100:.1f}%", (x1, y1-10),
                        font, 0.5, (255, 0, 0), 2)

    # === PROSES KAMERA KANAN (Deteksi + Segmentasi + Info) ===
    detection_right = frame_right.copy()
    segmentation_right = np.zeros_like(frame_right)
    right_detections = []  # List untuk menyimpan koordinat & maturity dari kamera kanan

    results_right = model(frame_right, device=device)
    for result in results_right:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for idx, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = map(int, box[:4])
            # Ekstrak ROI (region of interest) dari frame kanan
            roi = frame_right[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Konversi ROI ke HSV untuk segmentasi
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Definisikan rentang warna untuk cabai dan daun dalam HSV
            # red_lower1, red_upper1 = np.array([0, 50, 50]), np.array([10, 255, 255])
            red_lower2, red_upper2 = np.array([170, 50, 50]), np.array([180, 255, 255])
            orange_lower, orange_upper = np.array([10, 100, 100]), np.array([25, 255, 255])
            green_lower, green_upper = np.array([35, 30, 30]), np.array([85, 255, 150])
            leaf_lower, leaf_upper = np.array([35, 50, 50]), np.array([85, 255, 255])

            # Buat mask untuk masing-masing warna pada ROI
            # red_mask = cv2.inRange(roi_hsv, red_lower1, red_upper1) + cv2.inRange(roi_hsv, red_lower2, red_upper2)
            red_mask = cv2.inRange(roi_hsv, red_lower2, red_upper2)
            orange_mask = cv2.inRange(roi_hsv, orange_lower, orange_upper)
            green_mask = cv2.inRange(roi_hsv, green_lower, green_upper)
            leaf_mask = cv2.inRange(roi_hsv, leaf_lower, leaf_upper)

            # Gabungkan mask warna cabai dan hilangkan area daun
            combined_mask = (red_mask + orange_mask + green_mask) & ~leaf_mask

            # Hitung jumlah piksel tiap mask dalam ROI
            total_pixels = cv2.countNonZero(combined_mask)
            red_pixels = cv2.countNonZero(cv2.bitwise_and(red_mask, combined_mask))
            orange_pixels = cv2.countNonZero(cv2.bitwise_and(orange_mask, combined_mask))
            green_pixels = cv2.countNonZero(cv2.bitwise_and(green_mask, combined_mask))

            red_pct = (red_pixels / total_pixels * 100) if total_pixels > 0 else 0
            orange_pct = (orange_pixels / total_pixels * 100) if total_pixels > 0 else 0
            green_pct = (green_pixels / total_pixels * 100) if total_pixels > 0 else 0

            # Tentukan tingkat kematangan berdasarkan warna dominan
            if red_pct > orange_pct and red_pct > green_pct:
                maturity = "Matang Merah"
                box_color = (0, 0, 255)
            elif orange_pct > red_pct and orange_pct > green_pct:
                maturity = "Matang Oren"
                box_color = (0, 165, 255)
            else:
                maturity = "Belum Matang (Hijau)"
                box_color = (0, 255, 0)

            cv2.rectangle(detection_right, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(detection_right, f"{maturity} ({conf*100:.1f}%)",
                        (x1, y1-10), font, 0.5, box_color, 2)

            # Buat tampilan segmentasi untuk ROI (konversi mask ke citra 3-channel)
            mask_color = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            segmentation_right[y1:y2, x1:x2] = mask_color

            center_x_right = int((x1 + x2) / 2)
            center_y_right = int((y1 + y2) / 2)
            right_detections.append({
                'center_x': center_x_right,
                'center_y': center_y_right,
                'maturity': maturity
            })

    # === KOMBINASI INFORMASI DETEKSI KIRI & KANAN ===
    combined_info = []
    n = min(len(left_detections), len(right_detections))
    for i in range(n):
        left_det = left_detections[i]
        right_det = right_detections[i]
        distance = calculate_distance(left_det['center_x'], right_det['center_x'])

        info_text = (
            f"Objek {i + 1}:\n"
            f"{right_det['maturity']}\n"
            f"Kiri: ({left_det['center_x']}, {left_det['center_y']})\n"
            f"Kanan: ({right_det['center_x']}, {right_det['center_y']})\n"
            f"Jarak: {distance:.2f} cm"
        )
        combined_info.append(info_text)

    # Siapkan canvas info (background hitam) ukuran 640x480
    info_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    line_height = 20
    y_offset = line_height  # Mulai offset y
    for info in combined_info:
        # Bagi teks berdasarkan baris (\n)
        lines = info.split('\n')
        for line in lines:
            cv2.putText(info_canvas, line, (10, y_offset),
                        font, 0.6, (0, 255, 255), 2)
            y_offset += line_height

    # === SUSUN TAMPILAN OUTPUT ===
    # Baris atas: gabungkan deteksi dari kamera kiri dan kanan secara berdampingan (640+640 = 1280 x 480)
    top_row = np.hstack((detection_left, detection_right))
    # Baris bawah: kiri: info_canvas (640x480) dan kanan: segmentasi dari kamera kanan (640x480)
    bottom_row = np.hstack((info_canvas, segmentation_right))
    combined_output = np.vstack((top_row, bottom_row))  # Ukuran akhir: 1280 x 960

    cv2.imshow("Deteksi Cabai", combined_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
