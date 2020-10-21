import cv2
import depthai
import numpy as np

print('Using depthai module from: ', depthai.__file__, ' version: ', depthai.__version__)

print("Creating pipeline...")
p = depthai.Pipeline()

# ColorCamera
print("Creating Color Camera...")
cam = p.createColorCamera()
cam.setPreviewSize(1920, 1080)
cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setCamId(0)
cam_xout = p.createXLinkOut()
cam_xout.setStreamName("preview")
cam.preview.link(cam_xout.input)
print("Pipeline created.")

device = depthai.Device()
print("Starting pipeline...")
device.startPipeline(p)
preview = device.getOutputQueue("preview")
print("Pipeline started.")
faces = []
while True:
    raw_data = preview.get().getData()
    print(len(raw_data))
    preview_frame = np.array(raw_data).reshape((3, 1080, 1920)).transpose(1, 2, 0).astype(np.uint8)
    cv2.imshow("test", preview_frame)

    if cv2.waitKey(1) == ord("q"):
        break
