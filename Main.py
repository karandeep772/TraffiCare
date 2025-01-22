import torch
import cv2
import os
import numpy as np

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the YOLOv5 model
model_path = os.path.join(script_dir, 'Models', 'best2.pt')
model_path1 = os.path.join(script_dir, 'yolov5')
model = torch.hub.load('./yolov5', 'custom', path='./Models/best2.pt', source='local')
model.conf = 0.25  # Confidence threshold

# Define paths for video input and output
video_path = os.path.join(script_dir, 'V1.mp4')
output_path = os.path.join(script_dir, 'output_video.avi')

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for output file
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames are available

    # Convert frame (OpenCV uses BGR; model expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(frame_rgb)

    # Get detection results and store them in a list
    detections = []
    for result in results.xyxy[0]:  # xyxy contains detection data
        x1, y1, x2, y2, conf, cls = result.tolist()
        detections.append({
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'confidence': conf,
            'class': int(cls)
        })
    
    # Count number of objects detected
    num_objects = len(detections)

    # Render results on the frame
    annotated_frame = results.render()[0]  # Rendered frame is in RGB format

    # Convert back to BGR for OpenCV
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Display the number of objects on the frame
    cv2.putText(annotated_frame, f'Objects: {num_objects}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Optional: Display the frame (comment out for faster processing)
    cv2.imshow('YOLOv5 Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Print detections and count for the current frame (for debugging or analysis)
    print(f"Frame Detected Objects: {num_objects}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved to {output_path}")
