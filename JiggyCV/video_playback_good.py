import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import numpy as np
import sys

model_path = 'pose_landmarker_heavy.task'

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  # Convert to BGR before creating a copy for drawing
  bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
  annotated_image = np.copy(bgr_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image  # No need to convert again as it's already in BGR


base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# Check if video path is provided as command line argument
if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = '/Users/rahulnair/Desktop/JIGGY/videos/vid3.mp4'  # Default video path

# Open the reference video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    sys.exit(1)

print(f"Processing video: {video_path}")

# Get video properties for display
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")

frame_count = 0
last_detection_result = None  # Store the last detection result

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    annotated_image = frame.copy()
    
    # Process every 3rd frame but maintain original playback speed
    if frame_count % 4 == 0:  
        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect pose landmarks
        last_detection_result = detector.detect(mp_image)
    
    # Draw landmarks using the last available detection result
    if last_detection_result is not None:
        annotated_image = draw_landmarks_on_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), last_detection_result)
    
   
    # Resize if the video is too large for display
    if frame_width > 1280 or frame_height > 720:
        display_scale = min(1280 / frame_width, 720 / frame_height)
        display_width = int(frame_width * display_scale)
        display_height = int(frame_height * display_scale)
        annotated_image = cv2.resize(annotated_image, (display_width, display_height))
    
    cv2.imshow('Pose Detection', annotated_image)
    
    # Press 'q' to quit, space to pause/resume
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space bar to pause/resume
        print("Paused. Press space to continue.")
        while True:
            if cv2.waitKey(0) & 0xFF == ord(' '):
                break

cap.release()
cv2.destroyAllWindows()
print(f"Finished processing {frame_count} frames.")