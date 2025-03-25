import cv2
import numpy as np
import torch
from ultralytics import YOLOv10

# Cek apakah GPU tersedia, jika tidak pakai CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model YOLO ke perangkat (GPU/CPU)
model_path = "model-1.pt"
model = YOLOv10(model_path).to(device)

# Buka kamera (0 untuk kamera default)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Target resolusi untuk proses (lebih ringan dan cepat)
target_width, target_height = 960, 540

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah ukuran frame untuk proses deteksi
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # Salin frame untuk tampilan deteksi dan buat canvas untuk segmentasi
    detection_view = resized_frame.copy()
    segmentation_view = np.zeros_like(resized_frame)

    # Lakukan deteksi dengan YOLO pada frame yang sudah diresize
    results = model(resized_frame, device=device)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()  # Confidence score

        for box, confidence in zip(boxes, confidences):
            # Ambil koordinat bounding box (dalam skala frame yang sudah diresize)
            x1, y1, x2, y2 = map(int, box[:4])

            # Ekstrak region of interest (ROI) dari frame deteksi
            roi = resized_frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue  # Lewati jika ROI kosong

            # Konversi ROI ke ruang warna HSV
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Definisikan rentang warna untuk cabai dan daun dalam HSV
            # red_lower1, red_upper1 = np.array([0, 50, 50]), np.array([10, 255, 255])
            red_lower2, red_upper2 = np.array([170, 50, 50]), np.array([180, 255, 255])
            orange_lower, orange_upper = np.array([10, 100, 100]), np.array([25, 255, 255])
            green_lower, green_upper = np.array([35, 30, 30]), np.array([85, 255, 150])
            leaf_lower, leaf_upper = np.array([35, 50, 50]), np.array([85, 255, 255])

            # Buat mask untuk masing-masing warna di dalam ROI
            # red_mask_roi = cv2.inRange(roi_hsv, red_lower1, red_upper1) + cv2.inRange(roi_hsv, red_lower2, red_upper2)
            red_mask_roi = cv2.inRange(roi_hsv, red_lower2, red_upper2)
            orange_mask_roi = cv2.inRange(roi_hsv, orange_lower, orange_upper)
            green_mask_roi = cv2.inRange(roi_hsv, green_lower, green_upper)
            leaf_mask_roi = cv2.inRange(roi_hsv, leaf_lower, leaf_upper)

            # Gabungkan mask warna cabai dan hilangkan mask daun
            combined_mask_roi = (red_mask_roi + orange_mask_roi + green_mask_roi) & ~leaf_mask_roi

            # Hitung piksel dari masing-masing mask pada ROI
            total_pixels = cv2.countNonZero(combined_mask_roi)
            red_pixels = cv2.countNonZero(cv2.bitwise_and(red_mask_roi, combined_mask_roi))
            orange_pixels = cv2.countNonZero(cv2.bitwise_and(orange_mask_roi, combined_mask_roi))
            green_pixels = cv2.countNonZero(cv2.bitwise_and(green_mask_roi, combined_mask_roi))

            red_percentage = (red_pixels / total_pixels * 100) if total_pixels > 0 else 0
            orange_percentage = (orange_pixels / total_pixels * 100) if total_pixels > 0 else 0
            green_percentage = (green_pixels / total_pixels * 100) if total_pixels > 0 else 0

            # Tentukan tingkat kematangan berdasarkan warna dominan di dalam bounding box
            if red_percentage > orange_percentage and red_percentage > green_percentage:
                maturity = "Matang Merah"
                box_color = (0, 0, 255)  # Merah
            elif orange_percentage > red_percentage and orange_percentage > green_percentage:
                maturity = "Matang Oren"
                box_color = (0, 165, 255)  # Orange
            else:
                maturity = "Belum Matang (Hijau)"
                box_color = (0, 255, 0)  # Hijau

            # Gambar bounding box dan label pada tampilan deteksi
            cv2.rectangle(detection_view, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(detection_view, f"{maturity} ({confidence * 100:.1f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Buat tampilan segmentasi untuk ROI dengan mengonversi mask menjadi citra 3 channel
            mask_roi_color = cv2.cvtColor(combined_mask_roi, cv2.COLOR_GRAY2BGR)
            segmentation_view[y1:y2, x1:x2] = mask_roi_color

    # Gabungkan tampilan deteksi dan segmentasi secara berdampingan
    combined_view = np.hstack((detection_view, segmentation_view))
    cv2.imshow("Real-time Cabai Detection (Normal vs Segmentasi)", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
