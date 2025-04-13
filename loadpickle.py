import pickle
import csv
import os

# Load the pickle file
with open("pickleFolder/no_shirt_bball_angles.pkl", "rb") as f:
    loaded_pose_data = pickle.load(f)

# Create CSV output file
csv_file_path = "/Users/rahulnair/Desktop/JIGGY/anglesCSV/no_shirt_bball_angles.csv"

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