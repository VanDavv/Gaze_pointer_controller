import time
from pathlib import Path

import depthai
import cv2
import numpy as np

print('Using depthai module from: ', depthai.__file__, ' version: ', depthai.__version__)

print("Creating pipeline...")
p = depthai.Pipeline()

# ColorCamera
print("Creating Color Camera...")
cam = p.createColorCamera()
cam.setPreviewSize(300, 300)
cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setCamId(0)
cam_xout = p.createXLinkOut()
cam_xout.setStreamName("preview")
cam.preview.link(cam_xout.input)

# NeuralNetwork
print("Creating Face Detection Neural Network...")
face_nn = p.createNeuralNetwork()
face_nn.setBlobPath(str(Path("models/face-detection-retail-0004/face-detection-retail-0004.blob").resolve().absolute()))
face_nn_xout = p.createXLinkOut()
face_nn_xout.setStreamName("face_nn")
cam.preview.link(face_nn.input)
face_nn.out.link(face_nn_xout.input)

# NeuralNetwork
print("Creating Landmarks Detection Neural Network...")
land_nn = p.createNeuralNetwork()
land_nn.setBlobPath(
    str(Path("models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.blob").resolve().absolute())
)
land_nn_xin = p.createXLinkIn()
land_nn_xin.setStreamName("landmark_in")
land_nn_xin.out.link(land_nn.input)
land_nn_xout = p.createXLinkOut()
land_nn_xout.setStreamName("landmark_nn")
land_nn.out.link(land_nn_xout.input)
print("Pipeline created.")


# Get first device and connect
print("Looking for devices...")
found, deviceInfo = depthai.XLinkConnection.getFirstDevice(depthai.X_LINK_UNBOOTED)
if found:
    print("Device found.")
    device = depthai.Device(deviceInfo)
    print("Starting pipeline...")
    device.startPipeline(p)
    preview = device.getOutputQueue("preview")
    face_nn = device.getOutputQueue("face_nn")
    land_in = device.getInputQueue("landmark_in")
    land_nn = device.getOutputQueue("landmark_nn")
    print("Pipeline started.")
    faces = []
    while True:
        if face_nn.has():
            print("Fetching face_nn output")
            face_raw_output = np.frombuffer(bytes(face_nn.get().data), dtype=np.float16).reshape((200, 7))
            print(f"Received face_nn output, size: {face_raw_output.size}")
            faces = [
                {
                    "label": data[1],
                    "conf": data[2],
                    "x_min": data[3],
                    "y_min": data[4],
                    "x_max": data[5],
                    "y_max": data[6],
                }
                for data in face_raw_output
                if data[2] > 0.5
            ]
            print(f"Detected faces: {len(faces)}")

        if preview.has():
            print("Getting preview frame...")
            preview_frame = np.array(preview.get().data).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)
            preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            print("Preview frame received, cropping face_nn bbox...")

            for e in faces:
                face_frame = preview_frame[
                    int(e['y_min'] * 300):int(e['y_max'] * 300),
                    int(e['x_min'] * 300):int(e['x_max'] * 300),
                ]
                if face_frame.size == 0:
                    continue
                print(f"Face cropped, size: {face_frame.size}")
                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)
                print("Preparing land_nn input buffer...")
                buff = depthai.RawBuffer()
                buff.data = [val for channel in face_frame.transpose(2, 0, 1) for y_col in channel for val in y_col]
                print("Sending buffer to land_nn...")
                land_in.send(buff)
                print("Getting data from land_nn...")
                if not land_nn.has():
                    time.sleep(0.1)
                    if not land_nn.has():
                        continue
                landmarks = np.frombuffer(bytes(land_nn.get().data), dtype=np.float16).reshape((5, 2))
                h, w = face_frame.shape[:2]
                for x, y in landmarks:
                    cv2.circle(face_frame, (int(x * w), int(y * h)), 3, (255, 255, 0))
                cv2.imshow("face", face_frame)
            preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("test", preview_frame)

        if cv2.waitKey(1) == ord("q"):
            break
else:
    print('No depthai devices found...')
