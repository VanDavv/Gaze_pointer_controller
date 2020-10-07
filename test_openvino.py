import time
from pathlib import Path

import cv2
import numpy as np
from openvino.inference_engine import IECore

debug = True


def prepare_input(nnet, in_dict):
    result = {}
    for key in in_dict:
        shape = nnet.inputs[key].shape
        in_frame = np.array(in_dict[key])
        if len(shape) == 4:
            in_frame = cv2.resize(in_frame, tuple(shape[-2:]))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        result[key] = in_frame.reshape(shape)
    return result


def run_net(nnet, in_dict):
    nnet.start_async(request_id=0, inputs=prepare_input(nnet, in_dict))
    while nnet.requests[0].wait(-1) != 0:
        time.sleep(0.1)
    result = {
        key: nnet.requests[0].outputs[key][0]
        for key in nnet.requests[0].outputs
    }
    return result


cap = cv2.VideoCapture(str(Path("demo.mp4").resolve().absolute()))
ie = IECore()
nn_path = Path("models/face-detection-retail-0004")
definition = str(next(nn_path.glob("*.xml")).resolve().absolute())
weights = str(next(nn_path.glob("*.bin")).resolve().absolute())
net = ie.read_network(model=definition, weights=weights)
exec_net = ie.load_network(network=net, num_requests=0, device_name="MYRIAD")

while cap.isOpened():
    read_correctly, frame = cap.read()
    if not read_correctly:
        raise SystemExit(0)
    out = run_net(exec_net, {"data": frame})
    height, width = frame.shape[:2]
    coords = [
        (int(obj[3] * width), int(obj[4] * height), int(obj[5] * width), int(obj[6] * height))
        for obj in out["detection_out"][0]
        if obj[2] > 0.6
    ]
    head_image = frame[coords[0][1]:coords[0][3], coords[0][0]:coords[0][2]]
    print("HEAD_IMAGE_SIZE:", head_image.size)

