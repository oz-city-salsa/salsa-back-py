import gradio as gr
import mediapipe as mp
import cv2
import csv
import os
import tempfile

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    for idx, landmark in enumerate(landmarks):
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])

def process_video_with_pose_detection(video_path):
    if video_path is None:
        return None, None
    
    # Initialize MediaPipe Pose and Drawing utilities
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary files for output
    output_video_fd, output_video_path = tempfile.mkstemp(suffix='.mp4')
    output_csv_fd, output_csv_path = tempfile.mkstemp(suffix='.csv')
    
    # Close the file descriptors as we'll use the paths directly
    os.close(output_video_fd)
    os.close(output_csv_fd)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_number = 0
    csv_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        result = pose.process(frame_rgb)
        
        # Draw the pose landmarks on the frame
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Add the landmark coordinates to the CSV data
            write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data)
        
        # Write the frame to output video
        out.write(frame)
        frame_number += 1
    
    # Release resources
    cap.release()
    out.release()
    pose.close()
    
    # Save the CSV data
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
        csv_writer.writerows(csv_data)
    
    return output_video_path, output_csv_path

# Create Gradio interface
demo = gr.Interface(
    fn=process_video_with_pose_detection,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Video(label="Processed Video with Pose Landmarks"),
        gr.File(label="Download Pose Landmarks CSV")
    ],
    title="Video Pose Detection",
    description="Upload a video to detect and visualize human pose landmarks using MediaPipe. The processed video will show the detected pose landmarks, and you can download the landmark coordinates as a CSV file.",
    examples=[
        ["tst.mp4", "tst2.mp4", "tst3.mp4", "tst4.mp4", "tst5.mp4"] if os.path.exists("tst5.mp4") else []
    ]
)

if __name__ == "__main__":
    demo.launch()