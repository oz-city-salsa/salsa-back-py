#@markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import mediapipe as mp


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  
  # Ensure the image is in the correct format (3 channels, uint8)
  if len(rgb_image.shape) == 3:
    if rgb_image.shape[2] == 4:
      # Convert RGBA to RGB by removing alpha channel
      annotated_image = rgb_image[:, :, :3].copy()
    elif rgb_image.shape[2] == 3:
      annotated_image = np.copy(rgb_image)
    else:
      # Handle other channel counts
      annotated_image = np.copy(rgb_image)
  else:
    # Convert grayscale to RGB if needed
    if len(rgb_image.shape) == 2:
      annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
    else:
      annotated_image = np.copy(rgb_image)
  
  # Ensure the image is uint8
  if annotated_image.dtype != np.uint8:
    annotated_image = (annotated_image * 255).astype(np.uint8)

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
  return annotated_image

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
try:
    image = mp.Image.create_from_file("image.png")
    print(f"Image loaded successfully. Format: {image.image_format}")
    print(f"Image shape: {image.numpy_view().shape}")
    print(f"Image dtype: {image.numpy_view().dtype}")
except Exception as e:
    print(f"Error loading image: {e}")
    print("Creating sample image for pose detection")
    # Create a sample image with a simple figure
    sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple stick figure
    cv2.circle(sample_img, (320, 120), 30, (255, 255, 255), -1)  # head
    cv2.line(sample_img, (320, 150), (320, 300), (255, 255, 255), 5)  # body
    cv2.line(sample_img, (320, 200), (280, 250), (255, 255, 255), 5)  # left arm
    cv2.line(sample_img, (320, 200), (360, 250), (255, 255, 255), 5)  # right arm
    cv2.line(sample_img, (320, 300), (280, 380), (255, 255, 255), 5)  # left leg
    cv2.line(sample_img, (320, 300), (360, 380), (255, 255, 255), 5)  # right leg
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=sample_img)

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow('annotated_image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()