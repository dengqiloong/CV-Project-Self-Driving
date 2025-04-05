import torch
import cv2
import os

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best2.pt', force_reload=True)

# Load video
video_path = 'inference_videos/v1.mp4'
if not os.path.exists(video_path):
    print(f"❌ Video file does not exist: {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"✅ Video loaded: {video_path}")
print(f"Resolution: {width}x{height}, FPS: {fps}")

if width == 0 or height == 0 or fps == 0:
    print("❌ Invalid video properties. Check the video file.")
    exit()

# Output path
output_path = 'runs/detect/inference_output3.mp4'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Define VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    rendered_frame = results.render()[0]
    out.write(rendered_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Video saved to {output_path}")

