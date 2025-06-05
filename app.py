import numpy as np
import streamlit as st
from PIL import Image
import cv2
from ultralytics import YOLO
import torch
import time
from collections import defaultdict, deque

from sort import Sort  # Make sure sort.py is in the same folder

# Load YOLO model
model = YOLO("best.pt")  # Replace with your trained model path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Streamlit UI
st.title("Helmet Detection with Tracking")

# Sidebar
st.sidebar.header("Upload Video")
video_data = st.sidebar.file_uploader("Upload video", ['mp4','mov', 'avi'])
run_button = st.sidebar.button("Process")

# Temp file paths
input_path = './save.mp4'
output_path = './result.mp4'

# Write file helper
def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as f:
        f.write(bytesio.getbuffer())

# Smoothing memory
label_memory = defaultdict(lambda: deque(maxlen=5))  # Store last 5 labels per ID
track_labels = {}  # Final label per ID

if video_data and run_button:
    # Save uploaded video
    write_bytesio_to_file(input_path, video_data)

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, frame_fps, (width, height))

    st.write(f"Resolution: {width}x{height}, FPS: {frame_fps}")

    tracker = Sort()
    frame_times = []
    start_video = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()
        detections = []

        with torch.no_grad():
            results = model(frame)

        boxes = results[0].boxes
        if boxes:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])

                if conf > 0.5:
                    detections.append([x1, y1, x2, y2, conf])

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        tracks = tracker.update(detections)

        for i, track in enumerate(tracks):
            x1, y1, x2, y2, track_id = map(int, track)

            # Find matching detection class by proximity
            matched_cls = None
            for box in boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                iou = (min(x2, bx2) - max(x1, bx1)) * (min(y2, by2) - max(y1, by1))
                if iou > 0:
                    matched_cls = int(box.cls[0])
                    break

            if matched_cls is not None:
                label = model.names[matched_cls]
                label_memory[track_id].append(label)

                # Smooth label
                common_label = max(set(label_memory[track_id]), key=label_memory[track_id].count)
                track_labels[track_id] = common_label

                # Draw
                if common_label == "helmet" or common_label == 'helmets':
                    color = (0, 255, 0)
                elif common_label == "non_helmet" or common_label == 'non_helmets':
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{common_label} ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        frame_times.append(time.time() - frame_start)

    cap.release()
    out.release()

    total_time = time.time() - start_video
    avg_time = np.mean(frame_times)

    # Display results
    st.header("Original Video")
    st.video(input_path)
    st.header("Processed Video")
    st.video(output_path)
    st.write(f"Total processing time: {total_time:.2f}s")
    st.write(f"Average frame time: {avg_time:.4f}s")

else:
    st.write("Upload a video and press Process to start.")
