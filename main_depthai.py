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
cam.setInterleaved(True)
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
        face_raw_output = face_nn.get().data
        data = [
            {
                "conf": face_raw_output[i * 7],
                "label": face_raw_output[i * 7 + 1],
                "score": face_raw_output[i * 7 + 2],
                "x_min": face_raw_output[i * 7 + 3],
                "y_min": face_raw_output[i * 7 + 4],
                "x_max": face_raw_output[i * 7 + 5],
                "y_max": face_raw_output[i * 7 + 6],
            }
            for i in range(len(face_raw_output) // 7)
            if face_raw_output[i * 7] != -1
        ]
        print(len(data), data)

        preview_frame = preview.get()
        cv2.imshow("test", np.array(preview_frame.data).reshape((300, 300, 3)).astype(np.uint8))


        if cv2.waitKey(1) == ord("q"):
            break
else:
    print('No depthai devices found...')
