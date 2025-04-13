import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import numpy as np
import math
import pickle
import os

model_path = 'pose_landmarker_full.task'

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

# Function to calculate angle between three points
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
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert to degrees
    angle = np.degrees(angle)
    
    return angle

# Define key joint angles to track
def get_pose_angles(landmarks):
    """
    Calculate key angles from pose landmarks
    Returns a dictionary of angles
    """
    # Convert landmarks to numpy array for easier manipulation
    points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
    
    # Define angles to track (using MediaPipe Pose indices)
    # Right elbow angle (shoulder, elbow, wrist)
    right_elbow = calculate_angle(
        [points[11][0], points[11][1]],  # Right shoulder
        [points[13][0], points[13][1]],  # Right elbow
        [points[15][0], points[15][1]]   # Right wrist
    )
    
    # Left elbow angle
    left_elbow = calculate_angle(
        [points[12][0], points[12][1]],  # Left shoulder
        [points[14][0], points[14][1]],  # Left elbow
        [points[16][0], points[16][1]]   # Left wrist
    )
    
    # Right knee angle
    right_knee = calculate_angle(
        [points[23][0], points[23][1]],  # Right hip
        [points[25][0], points[25][1]],  # Right knee
        [points[27][0], points[27][1]]   # Right ankle
    )
    
    # Left knee angle
    left_knee = calculate_angle(
        [points[24][0], points[24][1]],  # Left hip
        [points[26][0], points[26][1]],  # Left knee
        [points[28][0], points[28][1]]   # Left ankle
    )
    
    # Right shoulder angle (elbow, shoulder, hip)
    right_shoulder = calculate_angle(
        [points[13][0], points[13][1]],  # Right elbow
        [points[11][0], points[11][1]],  # Right shoulder
        [points[23][0], points[23][1]]   # Right hip
    )
    
    # Left shoulder angle
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

# Function to compare current pose with reference pose
def compare_poses(current_angles, reference_angles):
    """
    Compare current pose angles with reference pose angles
    Returns a similarity score (0-100)
    """
    if not reference_angles or not current_angles:
        return 0
    
    # Calculate mean absolute error for each angle
    errors = []
    for joint in current_angles:
        if joint in reference_angles:
            error = abs(current_angles[joint] - reference_angles[joint])
            errors.append(error)
    
    if not errors:
        return 0
    
    # Calculate mean error
    mean_error = np.mean(errors)
    
    # Convert to a score (0-100)
    # Lower error = higher score, with a maximum error threshold of 30 degrees
    max_error = 30.0
    score = max(0, 100 - (mean_error / max_error * 100))
    
    return score

# Add function to load pickle file with landmarks
# Function to load pickle file with pre-calculated angles
def load_reference_angles(pickle_file):
    """Load pre-calculated angle data from pickle file"""
    if not os.path.exists(pickle_file):
        print(f"Error: Reference angle file not found: {pickle_file}")
        return None
    
    try:
        with open(pickle_file, 'rb') as f:
            angles_data = pickle.load(f)
        print(f"Loaded {len(angles_data)} frames of reference angles")
        return angles_data
    except Exception as e:
        print(f"Error loading reference angles: {e}")
        return None

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


# Reference pose angles (these would be loaded from a saved reference pose)
# For now, we'll initialize with empty values
reference_angles = {}
is_reference_set = False

# Add variables for video comparison
source_video_path = '/Users/rahulnair/Desktop/JIGGY/videos/vid3.mp4'  # Fixed typo in path
reference_pickle_path = '/Users/rahulnair/Desktop/JIGGY/vid3_angles.pkl'  # Path to the angle pickle file
source_video = None
source_frame_index = 0
comparison_scores = []  # Array to store all comparison scores
is_comparing_video = False
reference_angles_data = None  # Will store the loaded angle data
current_angle_index = 0  # Index to track position in reference angles
video_fps = 30  # Default FPS, will be updated when video is loaded
frame_skip = 2  # Process every nth frame for smoother playback

# Load webcam
cap = cv2.VideoCapture(0)

# Capture landmarks from webcam
webcam_landmarks = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect pose landmarks
    detection_result = detector.detect(mp_image)
    
    # Draw landmarks on the frame
    annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
    
    # Process source video if comparing
    if is_comparing_video and reference_angles_data is not None:
        # Open reference video if not already open
        if source_video is None:
            source_video = cv2.VideoCapture(source_video_path)
            if not source_video.isOpened():
                print(f"Error: Could not open reference video {source_video_path}")
                is_comparing_video = False
            else:
                # Get video properties for better playback
                video_fps = source_video.get(cv2.CAP_PROP_FPS)
                if video_fps <= 0:
                    video_fps = 30  # Fallback if FPS can't be determined
                print(f"Opened reference video: {source_video_path} (FPS: {video_fps})")
        
        # Read frame from reference video at native speed
        if source_video and source_video.isOpened():
            ret_ref, ref_frame = source_video.read()
            if not ret_ref:
                # Reset video to beginning when it ends
                source_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_ref, ref_frame = source_video.read()
            
            if ret_ref:
                # Display reference frame in its native resolution
                cv2.imshow('Reference Video', ref_frame)
        
        # Get the current reference frame's angles - directly from pre-calculated data
        if current_angle_index < len(reference_angles_data):
            ref_angle_data = reference_angles_data[current_angle_index]
            
            # Check if angles exist for this reference frame
            if ref_angle_data['angles'] is not None:
                # Use pre-calculated angles directly
                reference_angles = ref_angle_data['angles']
                is_reference_set = True
                
                # Display reference frame number
                cv2.putText(annotated_image, f"Reference Frame: {ref_angle_data['frame']}", 
                           (10, annotated_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Move to next frame at native speed
            current_angle_index += 1
        else:
            # End of reference data
            print("End of reference angles reached")
            is_comparing_video = False
            current_angle_index = 0
            # Close reference video
            if source_video and source_video.isOpened():
                source_video.release()
                source_video = None
            cv2.destroyWindow('Reference Video')
    
    # Calculate angles and compare with reference pose
    if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
        # Get angles from the first detected person
        current_angles = get_pose_angles(detection_result.pose_landmarks[0])
        
        # Store webcam landmarks if comparing
        if is_comparing_video and len(detection_result.pose_landmarks) > 0:
            # Store every 4th frame
            if source_frame_index % 4 == 0:
                # Convert landmarks to a more usable format
                frame_landmarks = []
                for landmark in detection_result.pose_landmarks[0]:
                    frame_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    })
                
                webcam_landmarks.append({
                    'frame': source_frame_index,
                    'landmarks': frame_landmarks
                })
            
            source_frame_index += 1
        
        # Display angles on the frame with larger font size
        y_pos = 40
        for joint, angle in current_angles.items():
            cv2.putText(annotated_image, f"{joint}: {angle:.1f} deg", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_pos += 30
        
        # Compare with reference pose if it's set
        if is_reference_set:
            score = compare_poses(current_angles, reference_angles)
            comparison_scores.append(score)  # Store the score
            
            # Display score with color based on value
            score_color = (0, 255, 0)  # Green for good scores
            if score < 70:
                score_color = (0, 165, 255)  # Orange for medium scores
            if score < 50:
                score_color = (0, 0, 255)  # Red for poor scores
                
            cv2.putText(annotated_image, f"Pose Score: {score:.1f}%", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, score_color, 2)
            y_pos += 40
    
    cv2.imshow('Pose Detection', annotated_image)
    
    # Keyboard controls
    wait_time = int(1000 / video_fps) if is_comparing_video and video_fps > 0 else 1
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Set current pose as reference
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            reference_angles = get_pose_angles(detection_result.pose_landmarks[0])
            is_reference_set = True
            print("Reference pose set!")
    elif key == ord('v'):
        # Toggle comparison with pickle file and video display
        if not is_comparing_video:
            # Load reference angles from pickle file
            reference_angles_data = load_reference_angles(reference_pickle_path)
            
            if reference_angles_data:
                is_comparing_video = True
                current_angle_index = 0
                source_frame_index = 0
                webcam_landmarks = []  # Reset webcam landmarks
                comparison_scores = []  # Reset scores
                print("Starting comparison with reference angles and video")
            else:
                print(f"Could not load reference angles from: {reference_pickle_path}")
        else:
            # Stop comparison
            is_comparing_video = False
            current_angle_index = 0
            print("Stopped comparison")
            
            # Close reference video
            if source_video and source_video.isOpened():
                source_video.release()
                source_video = None
            cv2.destroyWindow('Reference Video')
            
            # Save webcam landmarks if we have any
            if webcam_landmarks:
                webcam_pickle_path = '/Users/rahulnair/Desktop/JIGGY/webcam_landmarks.pkl'
                with open(webcam_pickle_path, 'wb') as f:
                    pickle.dump(webcam_landmarks, f)
                print(f"Saved {len(webcam_landmarks)} webcam landmarks to {webcam_pickle_path}")
    elif key == ord('y'):
        # Print all comparison scores
        if comparison_scores:
            print("\nComparison Scores:")
            for i, score in enumerate(comparison_scores):
                print(f"Frame {i+1}: {score:.1f}%")
            print(f"Average Score: {np.mean(comparison_scores):.1f}%")
            print(f"Min Score: {np.min(comparison_scores):.1f}%")
            print(f"Max Score: {np.max(comparison_scores):.1f}%")
        else:
            print("No comparison scores available")

# Clean up
cap.release()
if source_video and source_video.isOpened():
    source_video.release()
cv2.destroyAllWindows()