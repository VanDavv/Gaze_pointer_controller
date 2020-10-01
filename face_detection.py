import time
from pathlib import Path

from openvino.inference_engine import IENetwork, IECore
import cv2


class FaceDetection:
    def __init__(self, ie):
        self.model_weights = Path("models/face-detection-retail-0004/face-detection-retail-0004.bin").resolve().absolute()
        self.model_structure = self.model_weights.with_suffix('.xml')

        self.model = IENetwork(str(self.model_structure), str(self.model_weights))

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self.net = ie.read_network(model=str(self.model_structure), weights=str(self.model_weights))
        self.exec_net = ie.load_network(network=self.net, num_requests=0, device_name="MYRIAD")
        self.input_blob = next(iter(self.exec_net.inputs))
        self.output_blob = next(iter(self.exec_net.outputs))

    def predict(self, image):
        height, width = image.shape[:2]
        input_dict = {}
        n, c, h, w = self.input_shape
        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        input_dict[self.input_name] = in_frame
        self.exec_net.start_async(request_id=0, inputs=input_dict)
        while True:
            status = self.exec_net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(0.1)
        coords = []
        for obj in self.exec_net.requests[0].outputs[self.output_blob][0][0]:
            if obj[2] > 0.6:
                xmin = int(obj[3] * width)
                ymin = int(obj[4] * height)
                xmax = int(obj[5] * width)
                ymax = int(obj[6] * height)
                coords.append((xmin, ymin, xmax, ymax))

        head_image = image[coords[0][1]:coords[0][3], coords[0][0]:coords[0][2]]
        for obj in coords:
            cv2.rectangle(image, (obj[0], obj[1]), (obj[2], obj[3]), (10, 245, 10), 2)
        return coords, image, head_image
