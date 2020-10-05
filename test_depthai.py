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
print("Creating Neural Network...")
nn = p.createNeuralNetwork()
nn.setBlobPath(str(Path("models/face-detection-retail-0004/face-detection-retail-0004.blob").resolve().absolute()))
nn_xout = p.createXLinkOut()
nn_xout.setStreamName("face_nn")
cam.preview.link(nn.input)
nn.out.link(nn_xout.input)
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
    print("Pipeline started.")
    while True:
        face_raw_output = np.frombuffer(bytes(face_nn.get().data), dtype=np.float16).reshape((200, 7))
        data = [
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

        preview_frame = np.array(preview.get().data).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)
        preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)

        for e in data:
            pt1 = int(e['x_min'] * 300), int(e['y_min'] * 300)
            pt2 = int(e['x_max'] * 300), int(e['y_max'] * 300)
            cv2.rectangle(preview_frame, pt1, pt2, 100, 2)

        preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("test", preview_frame)


        if cv2.waitKey(1) == ord("q"):
            break
else:
    print('No depthai devices found...')
