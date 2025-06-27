import cv2
import numpy as np
import os
import sys
from openpose import pyopenpose as op

# Set up OpenPose
params = {
    "model_folder": "/openpose/models",
    "net_resolution": "-1x256",
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def run_pose_estimation(image_np: np.ndarray):
    datum = op.Datum()
    datum.cvInputData = image_np
    opWrapper.emplaceAndPop([datum])
    return datum.poseKeypoints.tolist() if datum.poseKeypoints is not None else []
