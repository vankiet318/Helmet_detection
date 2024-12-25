import numpy as np
import streamlit as st
from PIL import Image
import cv2
from ultralytics import YOLO
import subprocess
import torch
import time

# Load the YOLO model
model = YOLO("best.pt")  # Replace with the path to your trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
# Streamlit app
st.title("Helmet Detection")

# Sidebar for uploading image
st.sidebar.header("Control")
video_data = st.sidebar.file_uploader("Upload file", ['mp4','mov', 'avi'])
temp_file_to_save = './save.mp4'
temp_file_result  = './result.mp4'
run_button = st.sidebar.button("Process")

frame_times = []

def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

if video_data:
    if run_button:
        # save uploaded video to disc
        write_bytesio_to_file(temp_file_to_save, video_data)

        cap = cv2.VideoCapture(temp_file_to_save)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = cap.get(cv2.CAP_PROP_FPS)
        st.write(width, height, frame_fps)
        
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'XVID')
        out_mp4 = cv2.VideoWriter(temp_file_result, fourcc_mp4, frame_fps, (width, height),isColor = True)
    
        video_start = time.time()

        while True:
            ret,frame = cap.read()
            if not ret: break

            frame_start = time.time()

            # Detect and process frame
            with torch.no_grad():
                output = model(frame)
                        
            for box in output[0].boxes:
                x, y, x2, y2 = map(int,box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                if conf > 0.5:
                    label = f"{model.names[cls]}"  # Label text
                    color = (0, 0, 255) if cls == 0 else (0, 255, 0)  # Red for class 0, Green otherwise
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            out_mp4.write(frame)

            frame_end = time.time()

            frame_times.append(frame_end - frame_start)
        
        video_end = time.time()

        video_time = video_end - video_start

        ## Close video files
        out_mp4.release()
        cap.release()
        
        ## Show results
        st.header("Original Video")
        st.video(temp_file_to_save)
        st.header("Output")
        st.video(temp_file_result)

        mean_frame_time = np.mean(frame_times)

        st.write(f"Total video processing time: {video_time:.4f} seconds")
        st.write(f"Average frame processing time: {mean_frame_time:.4f} seconds")
        
else:
    st.write("Upload a video to get started.")