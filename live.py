import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import numpy as np
import math
import os

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

# Reference pose angles (these would be loaded from a saved reference pose)
# For now, we'll initialize with empty values
reference_angles = {}
is_reference_set = False
reference_image = None  # Will store the reference image
reference_annotated = None  # Will store the annotated reference image

# Add variables for comparison
comparison_scores = []  # Array to store all comparison scores
reference_image_path = "/Users/rahulnair/Desktop/JIGGY/reference_pose.jpg"  # Fixed path to reference image

cap = cv2.VideoCapture(0)

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
    
    # Display reference image if set
    if reference_annotated is not None:
        cv2.imshow('Reference Pose', reference_annotated)
    
    # Calculate angles and compare with reference pose
    if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
        # Get angles from the first detected person
        current_angles = get_pose_angles(detection_result.pose_landmarks[0])
        
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
    
    cv2.imshow('Pose Detection', annotated_image)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Set current pose as reference
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            reference_angles = get_pose_angles(detection_result.pose_landmarks[0])
            reference_annotated = annotated_image.copy()
            is_reference_set = True
            print("Reference pose set from webcam!")
    elif key == ord('l'):
        # Load reference image from specified path
        if os.path.exists(reference_image_path):
            # Load the image
            reference_image = cv2.imread(reference_image_path)
            if reference_image is not None:
                # Process reference image for landmarks
                ref_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
                ref_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ref_rgb)
                ref_detection = detector.detect(ref_mp_image)
                
                if ref_detection.pose_landmarks and len(ref_detection.pose_landmarks) > 0:
                    # Get angles from reference image
                    reference_angles = get_pose_angles(ref_detection.pose_landmarks[0])
                    reference_annotated = draw_landmarks_on_image(ref_rgb, ref_detection)
                    is_reference_set = True
                    print(f"Reference pose set from image: {reference_image_path}")
                    
                    # Display reference angles
                    print("\nReference Angles:")
                    for joint, angle in reference_angles.items():
                        print(f"{joint}: {angle:.1f} degrees")
                else:
                    print(f"No pose detected in reference image")
            else:
                print(f"Could not read image: {reference_image_path}")
        else:
            print(f"Image not found: {reference_image_path}")
    elif key == ord('c'):
        # Clear reference pose
        reference_angles = {}
        reference_image = None
        reference_annotated = None
        is_reference_set = False
        cv2.destroyWindow('Reference Pose')
        print("Reference pose cleared")
    elif key == ord('s'):
        # Save current frame as reference image
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            save_path = f"/Users/rahulnair/Desktop/JIGGY/reference_pose.jpg"
            cv2.imwrite(save_path, annotated_image)
            print(f"Current frame saved as reference image: {save_path}")
            
            # Also set as reference
            reference_angles = get_pose_angles(detection_result.pose_landmarks[0])
            reference_annotated = annotated_image.copy()
            is_reference_set = True
    elif key == ord('y'):
        # Print all comparison scores
        if comparison_scores:
            print("\nComparison Scores:")
            for i, score in enumerate(comparison_scores[-10:]):  # Show last 10 scores
                print(f"Frame {i+1}: {score:.1f}%")
            print(f"Average Score: {np.mean(comparison_scores):.1f}%")
            print(f"Min Score: {np.min(comparison_scores):.1f}%")
            print(f"Max Score: {np.max(comparison_scores):.1f}%")
        else:
            print("No comparison scores available")

# Clean up
cap.release()
cv2.destroyAllWindows()