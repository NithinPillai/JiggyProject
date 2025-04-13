import numpy as np
import pandas as pd
import pickle
import os
import sys

# Window size for moving average
WINDOW_SIZE = 5
# Threshold for similarity (in degrees)
SIMILARITY_THRESHOLD = 30

def load_reference_angles(csv_path):
    """
    Load reference angles from CSV file
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading reference angles: {e}")
        sys.exit(1)

def load_current_angles(pickle_path):
    """
    Load current angles from pickle file
    """
    try:
        with open(pickle_path, 'rb') as f:
            angles_data = pickle.load(f)
        return angles_data
    except Exception as e:
        print(f"Error loading current angles: {e}")
        sys.exit(1)

def calculate_moving_average(angles_list, joint_name):
    """
    Calculate moving average for a specific joint angle, ignoring 181 values
    """
    valid_angles = []
    moving_averages = {}
    
    for frame_data in angles_list:
        frame = frame_data['frame']
        angles = frame_data['angles']
        
        # Skip if angles is None
        if angles is None:
            moving_averages[frame] = None
            continue
        
        # Get current angle
        current_angle = angles.get(joint_name)
        
        # If angle is 181 (not visible), don't update the moving average
        if current_angle == 181:
            moving_averages[frame] = 181  # Mark as not visible
            continue
        
        # Add current angle to valid angles list
        valid_angles.append(current_angle)
        
        # Keep only the last WINDOW_SIZE angles
        if len(valid_angles) > WINDOW_SIZE:
            valid_angles.pop(0)
        
        # Calculate moving average
        if valid_angles:
            moving_averages[frame] = np.mean(valid_angles)
        else:
            moving_averages[frame] = None
    
    return moving_averages

def calculate_reference_moving_average(df, joint_name):
    """
    Calculate moving average for reference angles from DataFrame
    """
    valid_angles = []
    moving_averages = {}
    
    for _, row in df.iterrows():
        frame = row['frame']
        angle = row[joint_name]
        
        # If angle is 181 (not visible), don't update the moving average
        if angle == 181:
            moving_averages[frame] = 181  # Mark as not visible
            continue
        
        # Add current angle to valid angles list
        valid_angles.append(angle)
        
        # Keep only the last WINDOW_SIZE angles
        if len(valid_angles) > WINDOW_SIZE:
            valid_angles.pop(0)
        
        # Calculate moving average
        if valid_angles:
            moving_averages[frame] = np.mean(valid_angles)
        else:
            moving_averages[frame] = None
    
    return moving_averages

def compare_angles(current_ma, reference_ma, joint_name):
    """
    Compare current moving average with reference moving average
    """
    results = {}
    
    # Get all frames from current moving average
    for frame, current_avg in current_ma.items():
        print(frame, current_avg)
        # Skip if current average is None
        if current_avg is None:
            results[frame] = {
                'status': 'unknown',
                'error': None,
                'joint': joint_name
            }
            continue
        
        # If current angle is 181, it's not visible
        if current_avg == 181:
            results[frame] = {
                'status': 'not visible',
                'error': None,
                'joint': joint_name
            }
            continue
        
        # Check if frame exists in reference moving average
        if frame not in reference_ma:
            results[frame] = {
                'status': 'no reference',
                'error': None,
                'joint': joint_name
            }
            continue
        
        reference_avg = reference_ma[frame]
        
        # If reference angle is 181 or None, it's not visible
        if reference_avg == 181 or reference_avg is None:
            results[frame] = {
                'status': 'reference not visible',
                'error': None,
                'joint': joint_name
            }
            continue
        
        # Calculate absolute error
        error = abs(current_avg - reference_avg)
        
        # Check if error is within threshold
        if error <= SIMILARITY_THRESHOLD:
            status = 'similar'
        else:
            status = 'not similar'
        
        results[frame] = {
            'status': status,
            'error': error,
            'joint': joint_name,
            'current_avg': current_avg,
            'reference_avg': reference_avg
        }
    
    return results

def main():
    # Define paths
    reference_csv = "/Users/rahulnair/Desktop/JIGGY/anglesCSV/white_jersey_bball_angles.csv"
    
    # Check if pickle file is provided as command line argument
    if len(sys.argv) > 1:
        current_pickle = sys.argv[1]
    else:
        # Default to the pickle file generated by randomChecker.py
        current_pickle = "/Users/rahulnair/Desktop/JIGGY/pickleFolder/no_shirt_bball_angles.pkl"
    
    print(f"Comparing angles from {current_pickle} with reference {reference_csv}")
    
    # Load angles
    reference_df = load_reference_angles(reference_csv)
    current_angles = load_current_angles(current_pickle)
    
    # Joint names to compare
    joint_names = ['right_elbow', 'left_elbow', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder']
    
    # Calculate moving averages and compare
    all_results = {}
    
    for joint in joint_names:
        print(f"Processing {joint}...")
        
        # Calculate moving averages
        current_ma = calculate_moving_average(current_angles, joint)
        reference_ma = calculate_reference_moving_average(reference_df, joint)
        
        # Compare angles
        results = compare_angles(current_ma, reference_ma, joint)
        
        # Add to all results
        all_results[joint] = results
    
    # Print results
    # print("\nComparison Results:")
    # print("-" * 80)
    
    # # Get all frames from current angles
    # frames = sorted([data['frame'] for data in current_angles])
    
    # for frame in frames:
    #     print(f"\nFrame {frame}:")
        
    #     for joint in joint_names:
    #         if frame in all_results[joint]:
    #             result = all_results[joint][frame]
    #             status = result['status']
                
    #             if status == 'similar':
    #                 error = result['error']
    #                 current_avg = result['current_avg']
    #                 reference_avg = result['reference_avg']
    #                 print(f"  {joint}: {status} (error: {error:.2f}°, current: {current_avg:.2f}°, reference: {reference_avg:.2f}°)")
    #             else:
    #                 print(f"  {joint}: {status}")
    #         else:
    #             print(f"  {joint}: no data")

if __name__ == "__main__":
    main()