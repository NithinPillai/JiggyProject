import numpy as np
import os
import pickle
import sys

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

def compare_angle_pickle_files(file1_path, file2_path):
    """
    Compare two pickle files containing angle data and return similarity scores
    
    Args:
        file1_path: Path to first pickle file with angle data
        file2_path: Path to second pickle file with angle data
        
    Returns:
        Dictionary with average, min, max scores and frame-by-frame scores
    """
    # Load the angle data from pickle files
    try:
        with open(file1_path, 'rb') as f:
            angles1 = pickle.load(f)
        print(f"Loaded {len(angles1)} frames from {file1_path}")
    except Exception as e:
        print(f"Error loading {file1_path}: {e}")
        return None
    
    try:
        with open(file2_path, 'rb') as f:
            angles2 = pickle.load(f)
        print(f"Loaded {len(angles2)} frames from {file2_path}")
    except Exception as e:
        print(f"Error loading {file2_path}: {e}")
        return None
    
    # Ensure both files contain angle data in the expected format
    if not angles1 or not angles2:
        print("One or both files contain no data")
        return None
    
    # Check if the data is in the expected format
    # We expect a list of dictionaries, each with an 'angles' key
    # or a list of dictionaries with the angle keys directly
    
    # Determine the format of the first file
    if isinstance(angles1[0], dict) and 'angles' in angles1[0]:
        # Format: [{'frame': X, 'angles': {'right_elbow': Y, ...}}, ...]
        angles1 = [frame['angles'] for frame in angles1 if 'angles' in frame and frame['angles'] is not None]
    elif isinstance(angles1[0], dict) and any(key in angles1[0] for key in ['right_elbow', 'left_elbow', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder']):
        # Format: [{'right_elbow': Y, 'left_elbow': Z, ...}, ...]
        # Already in the right format
        pass
    else:
        print(f"Unexpected format in {file1_path}")
        return None
    
    # Determine the format of the second file
    if isinstance(angles2[0], dict) and 'angles' in angles2[0]:
        # Format: [{'frame': X, 'angles': {'right_elbow': Y, ...}}, ...]
        angles2 = [frame['angles'] for frame in angles2 if 'angles' in frame and frame['angles'] is not None]
    elif isinstance(angles2[0], dict) and any(key in angles2[0] for key in ['right_elbow', 'left_elbow', 'right_knee', 'left_knee', 'right_shoulder', 'left_shoulder']):
        # Format: [{'right_elbow': Y, 'left_elbow': Z, ...}, ...]
        # Already in the right format
        pass
    else:
        print(f"Unexpected format in {file2_path}")
        return None
    
    # Filter out any None values that might have slipped through
    angles1 = [angle for angle in angles1 if angle is not None]
    angles2 = [angle for angle in angles2 if angle is not None]
    
    if not angles1 or not angles2:
        print("After filtering, one or both angle lists are empty")
        return None
    
    # Calculate similarity scores for each frame pair
    # If the files have different numbers of frames, we'll use the shorter one
    num_frames = min(len(angles1), len(angles2))
    scores = []
    
    print(f"Comparing {num_frames} frames...")
    
    for i in range(num_frames):
        try:
            score = compare_poses(angles1[i], angles2[i])
            scores.append(score)
            if i % 100 == 0:  # Print progress every 100 frames
                print(f"Processed {i}/{num_frames} frames...")
        except Exception as e:
            print(f"Error comparing frame {i}: {e}")
            print(f"Frame 1 data: {angles1[i]}")
            print(f"Frame 2 data: {angles2[i]}")
    
    if not scores:
        print("No valid scores calculated")
        return None
    
    # Calculate statistics
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # Print summary
    print(f"\nComparison Results:")
    print(f"Compared {len(scores)} frames")
    print(f"Average Score: {avg_score:.1f}%")
    print(f"Min Score: {min_score:.1f}%")
    print(f"Max Score: {max_score:.1f}%")
    
    # Save the scores to a file
    output_path = '/Users/rahulnair/Desktop/JIGGY/comparison_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({
            'average_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'scores': scores,
            'reference_file': file1_path,
            'input_file': file2_path
        }, f)
    print(f"Results saved to {output_path}")
    
    # Return results
    return {
        'average_score': avg_score,
        'min_score': min_score,
        'max_score': max_score,
        'scores': scores
    }

def visualize_scores(scores):
    """
    Visualize the similarity scores
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(scores)
        plt.title('Pose Similarity Scores')
        plt.xlabel('Frame')
        plt.ylabel('Similarity Score (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        
        # Save the plot
        plot_path = '/Users/rahulnair/Desktop/JIGGY/similarity_scores.png'
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        # Show the plot
        plt.show()
    except ImportError:
        print("Matplotlib not installed. Cannot visualize scores.")
    except Exception as e:
        print(f"Error visualizing scores: {e}")

# Main function to run the comparison
def main():
    # Define default paths
    default_reference_path = '/Users/rahulnair/Desktop/JIGGY/vid3_angles.pkl'
    default_input_path = '/Users/rahulnair/Desktop/JIGGY/vidcompare_angles2.pkl'
    
    if len(sys.argv) == 3:
        # If called with two arguments, use them as file paths
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    else:
        # Use default paths
        file1 = default_reference_path
        file2 = default_input_path
        print(f"Using default paths:")
        print(f"Reference: {file1}")
        print(f"Input: {file2}")
    
    # Check if files exist
    if not os.path.exists(file1):
        print(f"Error: Reference file not found at {file1}")
        return
    if not os.path.exists(file2):
        print(f"Error: Input file not found at {file2}")
        return
    
    # Run comparison
    results = compare_angle_pickle_files(file1, file2)
    
    # Visualize results if available
    if results and 'scores' in results:
        visualize_scores(results['scores'])

# Run the main function when the script is executed
if __name__ == "__main__":
    main()