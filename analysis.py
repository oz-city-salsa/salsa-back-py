import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from collections import defaultdict

def load_csv_data(filename):
    """Load CSV data and return as pandas DataFrame"""
    return pd.read_csv(filename)

def calculate_euclidean_distance(point1, point2):
    """Calculate euclidean distance between two 3D points"""
    return euclidean([point1['x'], point1['y'], point1['z']], 
                    [point2['x'], point2['y'], point2['z']])

def analyze_pose_differences(file1, file2, file3):
    """
    Analyze differences between pose data from three CSV files
    Returns distance matrices and statistics
    """
    # Load the data
    df1 = load_csv_data(file1)
    df2 = load_csv_data(file2)
    df3 = load_csv_data(file3)
    
    print(f"Loaded data:")
    print(f"  {file1}: {len(df1)} rows, {len(df1['frame_number'].unique())} frames")
    print(f"  {file2}: {len(df2)} rows, {len(df2['frame_number'].unique())} frames")
    print(f"  {file3}: {len(df3)} rows, {len(df3['frame_number'].unique())} frames")
    
    # Get common frames and landmarks
    frames1 = set(df1['frame_number'].unique())
    frames2 = set(df2['frame_number'].unique())
    frames3 = set(df3['frame_number'].unique())
    common_frames = frames1.intersection(frames2).intersection(frames3)
    
    landmarks1 = set(df1['landmark'].unique())
    landmarks2 = set(df2['landmark'].unique())
    landmarks3 = set(df3['landmark'].unique())
    common_landmarks = landmarks1.intersection(landmarks2).intersection(landmarks3)
    
    print(f"\nCommon frames: {len(common_frames)}")
    print(f"Common landmarks: {len(common_landmarks)}")
    print(f"Landmarks: {sorted(list(common_landmarks))}")
    
    # Filter data to only include common frames and landmarks
    df1_filtered = df1[(df1['frame_number'].isin(common_frames)) & 
                       (df1['landmark'].isin(common_landmarks))].sort_values(['frame_number', 'landmark'])
    df2_filtered = df2[(df2['frame_number'].isin(common_frames)) & 
                       (df2['landmark'].isin(common_landmarks))].sort_values(['frame_number', 'landmark'])
    df3_filtered = df3[(df3['frame_number'].isin(common_frames)) & 
                       (df3['landmark'].isin(common_landmarks))].sort_values(['frame_number', 'landmark'])
    
    # Calculate distances between files
    distances_1_2 = []  # me.csv vs me2.csv
    distances_1_3 = []  # me.csv vs me3.csv
    distances_2_3 = []  # me2.csv vs me3.csv
    
    # Distance by landmark
    landmark_distances_1_2 = defaultdict(list)
    landmark_distances_1_3 = defaultdict(list)
    landmark_distances_2_3 = defaultdict(list)
    
    # Distance by frame
    frame_distances_1_2 = defaultdict(list)
    frame_distances_1_3 = defaultdict(list)
    frame_distances_2_3 = defaultdict(list)
    
    print("\nCalculating distances...")
    
    for frame in sorted(common_frames):
        for landmark in sorted(common_landmarks):
            # Get corresponding points
            point1 = df1_filtered[(df1_filtered['frame_number'] == frame) & 
                                 (df1_filtered['landmark'] == landmark)].iloc[0]
            point2 = df2_filtered[(df2_filtered['frame_number'] == frame) & 
                                 (df2_filtered['landmark'] == landmark)].iloc[0]
            point3 = df3_filtered[(df3_filtered['frame_number'] == frame) & 
                                 (df3_filtered['landmark'] == landmark)].iloc[0]
            
            # Calculate distances
            dist_1_2 = calculate_euclidean_distance(point1, point2)
            dist_1_3 = calculate_euclidean_distance(point1, point3)
            dist_2_3 = calculate_euclidean_distance(point2, point3)
            
            # Store distances
            distances_1_2.append(dist_1_2)
            distances_1_3.append(dist_1_3)
            distances_2_3.append(dist_2_3)
            
            # Store by landmark
            landmark_distances_1_2[landmark].append(dist_1_2)
            landmark_distances_1_3[landmark].append(dist_1_3)
            landmark_distances_2_3[landmark].append(dist_2_3)
            
            # Store by frame
            frame_distances_1_2[frame].append(dist_1_2)
            frame_distances_1_3[frame].append(dist_1_3)
            frame_distances_2_3[frame].append(dist_2_3)
    
    return {
        'distances_1_2': distances_1_2,
        'distances_1_3': distances_1_3,
        'distances_2_3': distances_2_3,
        'landmark_distances_1_2': dict(landmark_distances_1_2),
        'landmark_distances_1_3': dict(landmark_distances_1_3),
        'landmark_distances_2_3': dict(landmark_distances_2_3),
        'frame_distances_1_2': dict(frame_distances_1_2),
        'frame_distances_1_3': dict(frame_distances_1_3),
        'frame_distances_2_3': dict(frame_distances_2_3),
        'common_frames': sorted(common_frames),
        'common_landmarks': sorted(common_landmarks)
    }

def print_summary_statistics(results):
    """Print summary statistics of the distance analysis"""
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall statistics
    print("\nOVERALL EUCLIDEAN DISTANCES:")
    print("-" * 40)
    
    stats = [
        ("me.csv vs me2.csv", results['distances_1_2']),
        ("me.csv vs me3.csv", results['distances_1_3']),
        ("me2.csv vs me3.csv", results['distances_2_3'])
    ]
    
    for name, distances in stats:
        print(f"\n{name}:")
        print(f"  Mean distance: {np.mean(distances):.4f}")
        print(f"  Median distance: {np.median(distances):.4f}")
        print(f"  Std deviation: {np.std(distances):.4f}")
        print(f"  Min distance: {np.min(distances):.4f}")
        print(f"  Max distance: {np.max(distances):.4f}")
        print(f"  Total points compared: {len(distances)}")
    
    # Per-landmark statistics
    print("\n\nPER-LANDMARK AVERAGE DISTANCES:")
    print("-" * 50)
    
    landmark_stats = {}
    for landmark in results['common_landmarks']:
        landmark_stats[landmark] = {
            'me_vs_me2': np.mean(results['landmark_distances_1_2'][landmark]),
            'me_vs_me3': np.mean(results['landmark_distances_1_3'][landmark]),
            'me2_vs_me3': np.mean(results['landmark_distances_2_3'][landmark])
        }
    
    # Sort landmarks by average distance across all comparisons
    sorted_landmarks = sorted(landmark_stats.items(), 
                            key=lambda x: np.mean(list(x[1].values())), 
                            reverse=True)
    
    print(f"{'Landmark':<20} {'me vs me2':<12} {'me vs me3':<12} {'me2 vs me3':<12} {'Average':<12}")
    print("-" * 80)
    
    for landmark, stats in sorted_landmarks:
        avg_dist = np.mean(list(stats.values()))
        print(f"{landmark:<20} {stats['me_vs_me2']:<12.4f} {stats['me_vs_me3']:<12.4f} "
              f"{stats['me2_vs_me3']:<12.4f} {avg_dist:<12.4f}")
    
    # Frame-based analysis (show first 10 frames as example)
    print("\n\nPER-FRAME AVERAGE DISTANCES (first 10 frames):")
    print("-" * 60)
    
    print(f"{'Frame':<8} {'me vs me2':<12} {'me vs me3':<12} {'me2 vs me3':<12} {'Average':<12}")
    print("-" * 60)
    
    for frame in sorted(results['common_frames'])[:10]:
        avg_1_2 = np.mean(results['frame_distances_1_2'][frame])
        avg_1_3 = np.mean(results['frame_distances_1_3'][frame])
        avg_2_3 = np.mean(results['frame_distances_2_3'][frame])
        overall_avg = np.mean([avg_1_2, avg_1_3, avg_2_3])
        
        print(f"{frame:<8} {avg_1_2:<12.4f} {avg_1_3:<12.4f} {avg_2_3:<12.4f} {overall_avg:<12.4f}")

def create_visualizations(results):
    """Create visualizations of the distance analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pose Data Comparison Analysis', fontsize=16)
    
    # 1. Overall distance distributions
    ax1 = axes[0, 0]
    ax1.hist(results['distances_1_2'], bins=50, alpha=0.7, label='me vs me2', color='blue')
    ax1.hist(results['distances_1_3'], bins=50, alpha=0.7, label='me vs me3', color='red')
    ax1.hist(results['distances_2_3'], bins=50, alpha=0.7, label='me2 vs me3', color='green')
    ax1.set_xlabel('Euclidean Distance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Euclidean Distances')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average distances per landmark
    ax2 = axes[0, 1]
    landmarks = results['common_landmarks']
    avg_distances_1_2 = [np.mean(results['landmark_distances_1_2'][lm]) for lm in landmarks]
    avg_distances_1_3 = [np.mean(results['landmark_distances_1_3'][lm]) for lm in landmarks]
    avg_distances_2_3 = [np.mean(results['landmark_distances_2_3'][lm]) for lm in landmarks]
    
    x = np.arange(len(landmarks))
    width = 0.25
    
    ax2.bar(x - width, avg_distances_1_2, width, label='me vs me2', alpha=0.8)
    ax2.bar(x, avg_distances_1_3, width, label='me vs me3', alpha=0.8)
    ax2.bar(x + width, avg_distances_2_3, width, label='me2 vs me3', alpha=0.8)
    
    ax2.set_xlabel('Landmarks')
    ax2.set_ylabel('Average Euclidean Distance')
    ax2.set_title('Average Distance per Landmark')
    ax2.set_xticks(x)
    ax2.set_xticklabels(landmarks, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distance over time (frames)
    ax3 = axes[1, 0]
    frames = sorted(results['common_frames'])[:50]  # Show first 50 frames
    frame_avg_1_2 = [np.mean(results['frame_distances_1_2'][f]) for f in frames]
    frame_avg_1_3 = [np.mean(results['frame_distances_1_3'][f]) for f in frames]
    frame_avg_2_3 = [np.mean(results['frame_distances_2_3'][f]) for f in frames]
    
    ax3.plot(frames, frame_avg_1_2, label='me vs me2', marker='o', markersize=3)
    ax3.plot(frames, frame_avg_1_3, label='me vs me3', marker='s', markersize=3)
    ax3.plot(frames, frame_avg_2_3, label='me2 vs me3', marker='^', markersize=3)
    
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('Average Euclidean Distance')
    ax3.set_title('Average Distance Over Time (First 50 Frames)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot of distances by comparison
    ax4 = axes[1, 1]
    data_to_plot = [results['distances_1_2'], results['distances_1_3'], results['distances_2_3']]
    labels = ['me vs me2', 'me vs me3', 'me2 vs me3']
    
    bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('Euclidean Distance')
    ax4.set_title('Distance Distribution Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pose_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    print("Starting pose data analysis...")
    
    # Analyze the three CSV files
    results = analyze_pose_differences('me.csv', 'me2.csv', 'me3.csv')
    
    # Print detailed statistics
    print_summary_statistics(results)
    
    # Create visualizations
    create_visualizations(results)
    
    print(f"\nAnalysis complete! Visualization saved as 'pose_comparison_analysis.png'")
    
    # Save detailed results to CSV
    detailed_results = []
    for frame in results['common_frames']:
        for landmark in results['common_landmarks']:
            idx = results['common_landmarks'].index(landmark)
            frame_idx = sorted(results['common_frames']).index(frame)
            base_idx = frame_idx * len(results['common_landmarks']) + idx
            
            detailed_results.append({
                'frame': frame,
                'landmark': landmark,
                'distance_me_vs_me2': results['distances_1_2'][base_idx],
                'distance_me_vs_me3': results['distances_1_3'][base_idx],
                'distance_me2_vs_me3': results['distances_2_3'][base_idx]
            })
    
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv('detailed_distance_analysis.csv', index=False)
    print("Detailed results saved to 'detailed_distance_analysis.csv'")

if __name__ == "__main__":
    main()
