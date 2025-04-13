from weakref import ref
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import numpy as np
import sys
import pickle
import os
import pandas as pd
import math
import csv


base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,  # Disable unused feature
    num_poses=1)  # Limit to one pose
detector = vision.PoseLandmarker.create_from_options(options)

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (landmarks)
    Args:
        a: first point [x, y]
        b: mid point [x, y] (the point at which we calculate the angle)
        c: end point [x, y]
    Returns:
        Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # print(a)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert to degrees
    angle = np.degrees(angle)
    
    return angle

def get_pose_angles(landmarks):
    """
    Calculate key angles from pose landmarks
    Returns a dictionary of angles
    """
    # Convert landmarks to numpy array for easier manipulation
    points = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks])
    
    # Define angles to track (using MediaPipe Pose indices)
    # Right elbow angle (shoulder, elbow, wrist)

    # points[x] refers to x, y, z, and visibility of landmark
    if (points[11][3] < 0.8 and points[13][3] < 0.8 or points[15][3] < 0.8):
        right_elbow = 181
    else: 
        right_elbow = calculate_angle(
            [points[11][0], points[11][1]],  # Right shoulder
            [points[13][0], points[13][1]],  # Right elbow
            [points[15][0], points[15][1]]   # Right wrist
        )
    
    # Left elbow angle
    if (points[12][3] < 0.8 or points[14][3] < 0.8 or points[16][3] < 0.8):
        left_elbow = 181
    else:
        left_elbow = calculate_angle(
            [points[12][0], points[12][1]],  # Left shoulder
            [points[14][0], points[14][1]],  # Left elbow
            [points[16][0], points[16][1]]   # Left wrist
        )
    
    # Right knee angle
    if (points[23][3] < 0.8 or points[25][3] < 0.8 or points[27][3] < 0.8):
        right_knee = 181
    else:
        right_knee = calculate_angle(
            [points[23][0], points[23][1]],  # Right hip
            [points[25][0], points[25][1]],  # Right knee
            [points[27][0], points[27][1]]   # Right ankle
        )
    
    # Left knee angle
    if (points[24][3] < 0.8 or points[26][3] < 0.8 or points[28][3] < 0.8):
        left_knee = 181
    else:
        left_knee = calculate_angle(
            [points[24][0], points[24][1]],  # Left hip
            [points[26][0], points[26][1]],  # Left knee
            [points[28][0], points[28][1]]   # Left ankle
        )
    
    # Right shoulder angle
    if (points[13][3] < 0.8 or points[11][3] < 0.8 or points[23][3] < 0.8):
        right_shoulder = 181
    else:
        right_shoulder = calculate_angle(
            [points[13][0], points[13][1]],  # Right elbow
            [points[11][0], points[11][1]],  # Right shoulder
            [points[23][0], points[23][1]]   # Right hip
        )
    
    # Left shoulder angle
    if (points[14][3] < 0.8 or points[12][3] < 0.8 or points[24][3] < 0.8):
        left_shoulder = 181
    else:
        left_shoulder = calculate_angle(
            [points[14][0], points[14][1]],  # Left elbow
            [points[12][0], points[12][1]],  # Left shoulder
            [points[24][0], points[24][1]]   # Left hip
        )
    
    return {
        'right_elbow': right_elbow,
        'left_elbow': left_elbow,
        'right_knee': right_knee,
        'left_knee': left_knee,
        'right_shoulder': right_shoulder,
        'left_shoulder': left_shoulder
    }

# Check if video path is provided as command line argument
if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = '/Users/rahulnair/Desktop/JIGGY/videos/icyhot.mov'  # Default video path

# Create output filename based on input video
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_file = f"/Users/rahulnair/Desktop/JIGGY/pickleFolder/{video_name}_angles.pkl"  # Changed to _angles.pkl

# Open the reference video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    sys.exit(1)

print(f"Processing video: {video_path}")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")

# Calculate processing resolution for faster processing
process_scale = 0.75  # Process at half resolution
process_width = int(frame_width * process_scale)
process_height = int(frame_height * process_scale)

# Array to store all angle data
all_angles = []  # Changed from all_landmarks to all_angles
frame_count = 0

moving_average = {
            'right_elbow': 0,
            'right_elbow_avg': 0,
            'left_elbow': 0,
            'left_elbow_avg': 0,
            'right_knee': 0,
            'right_knee_avg': 0,
            'left_knee': 0,
            'left_knee_avg': 0,
            'right_shoulder': 0,
            'right_shoulder_avg': 0,
            'left_shoulder': 0,
            'left_shoulder_avg': 0
        }

df = pd.read_csv('./anglesCSV/no_shirt_bball_angles.csv')
reference_data = df.to_numpy()
reference_data_iterator = 0;

# Process the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process every 4th frame to speed up processing
    if frame_count % 4 == 0:
        # Resize frame for faster processing
        process_frame = cv2.resize(frame, (process_width, process_height))
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect pose landmarks
        detection_result = detector.detect(mp_image)
        
        
        # Calculate and store angles if landmarks detected
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            # Calculate angles directly
            angles = get_pose_angles(detection_result.pose_landmarks[0])
            # {'right_elbow': 181, 'left_elbow': 181, 'right_knee': 181, 'left_knee': 181, 'right_shoulder': 181, 'left_shoulder': 181}

            # Right elbow
            res = {
                'right_elbow': angles['right_elbow'],
                'right_elbow_avg': moving_average['right_elbow_avg'],
                'left_elbow': angles['left_elbow'],
                'left_elbow_avg': moving_average['left_elbow_avg'],
                'right_knee': angles['right_knee'],
                'right_knee_avg': moving_average['right_knee_avg'],
                'left_knee': angles['left_knee'],
                'left_knee_avg': moving_average['left_knee_avg'],
                'right_shoulder': angles['right_shoulder'],
                'right_shoulder_avg': moving_average['right_shoulder_avg'],
                'left_shoulder': angles['left_shoulder'],
                'left_shoulder_avg': moving_average['left_shoulder_avg']
            }

            if angles['right_elbow'] != 181:
                if moving_average['right_elbow_avg'] != 0:
                    moving_average['right_elbow_avg'] = (moving_average['right_elbow_avg'] + angles['right_elbow']) / 2
                else:
                    moving_average['right_elbow_avg'] = angles['right_elbow']

            res['right_elbow_avg'] = moving_average['right_elbow_avg']

            # Left elbow
            if angles['left_elbow'] != 181:
                if moving_average['left_elbow_avg'] != 0:
                    moving_average['left_elbow_avg'] = (moving_average['left_elbow_avg'] + angles['left_elbow']) / 2
                else:
                    moving_average['left_elbow_avg'] = angles['left_elbow']
 
            res['left_elbow_avg'] = moving_average['left_elbow_avg']

            #Right knee
            if angles['right_knee'] != 181:
                if moving_average['right_knee_avg'] != 0:
                    moving_average['right_knee_avg'] = (moving_average['right_knee_avg'] + angles['right_knee']) / 2
                else:
                    moving_average['right_knee_avg'] = angles['right_knee']

            res['right_knee_avg'] = moving_average['right_knee_avg']

            #Left knee
            if angles['left_knee'] != 181:
                if moving_average['left_knee_avg'] != 0:
                    moving_average['left_knee_avg'] = (moving_average['left_knee_avg'] + angles['left_knee']) / 2
                else:
                    moving_average['left_knee_avg'] = angles['left_knee']

            res['left_knee_avg'] = moving_average['left_knee_avg']

            #Right shoulder
            if angles['right_shoulder'] != 181:
                if moving_average['right_shoulder_avg'] != 0:
                    moving_average['right_shoulder_avg'] = (moving_average['right_shoulder_avg'] + angles['right_shoulder']) / 2
                else:
                    moving_average['right_shoulder_avg'] = angles['right_shoulder']

            res['right_shoulder_avg'] = moving_average['right_shoulder_avg']

            #Left shoulder
            if angles['left_shoulder'] != 181:
                if moving_average['left_shoulder_avg'] != 0:
                    moving_average['left_shoulder_avg'] = (moving_average['left_shoulder_avg'] + angles['left_shoulder']) / 2
                else:
                    moving_average['left_shoulder_avg'] = angles['left_shoulder']

            res['left_shoulder_avg'] = moving_average['left_shoulder_avg']

            all_angles.append({
                'frame': frame_count,
                'angles': res.copy()
            })
        else:
            all_angles.append({
                'frame': frame_count,
                'angles': None
            })
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

with open(output_file, 'wb') as f:
    pickle.dump(all_angles, f)

with open("pickleFolder/no_shirt_bball_angles.pkl", "rb") as f:
    loaded_pose_data = pickle.load(f)

# Create CSV output file
csv_file_path = f"/Users/rahulnair/Desktop/JIGGY/anglesCSV/{video_name}_angles.csv"

# Open CSV file for writing
with open(csv_file_path, 'w', newline='') as csv_file:
    # Create CSV writer
    csv_writer = csv.writer(csv_file)
    
    # Write header row - updated for angle data
    header = ['frame', 'right_elbow', 'right_elbow_avg', 'left_elbow', 'left_elbow_avg', 'right_knee', 'right_knee_avg', 'left_knee', 'left_knee_avg', 'right_shoulder', 'right_shoulder_avg', 'left_shoulder', 'left_shoulder_avg']
    csv_writer.writerow(header)
    
    # Write data rows
    for frame_data in loaded_pose_data:
        frame_num = frame_data['frame']
        angles = frame_data['angles']
        
        if angles:  # Check if angles exist for this frame
            row = [
                frame_num,
                angles['right_elbow'],
                angles['right_elbow_avg'],
                angles['left_elbow'],
                angles['left_elbow_avg'],
                angles['right_knee'],
                angles['right_knee_avg'],
                angles['left_knee'],
                angles['left_knee_avg'],
                angles['right_shoulder'],
                angles['right_shoulder_avg'],
                angles['left_shoulder'],
                angles['left_shoulder_avg']
            ]
            csv_writer.writerow(row)

print(f"Angle data exported to CSV: {csv_file_path}")
print(f"Total frames with angle data: {len([f for f in loaded_pose_data if f['angles'] is not None])}")
