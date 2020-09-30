import time
from pathlib import Path

from openvino.inference_engine import IENetwork, IECore
import cv2


class FaceDetection:
    def __init__(self):
        self.model_weights = Path("models/face-detection-retail-0004/face-detection-retail-0004.bin").resolve().absolute()
        self.model_structure = self.model_weights.with_suffix('.xml')

        self.model = IENetwork(str(self.model_structure), str(self.model_weights))

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self, ie):
        self.net = ie.read_network(model=str(self.model_structure), weights=str(self.model_weights))
        self.exec_net = ie.load_network(network=self.net, num_requests=0, device_name="MYRIAD")
        self.input_blob = next(iter(self.exec_net.inputs))
        self.output_blob = next(iter(self.exec_net.outputs))

    def predict(self, image):
        feed_dict = self.preprocess_input(image)
        self.exec_net.start_async(request_id=0, inputs=feed_dict)
        while True:
            status = self.exec_net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
        coords = self.preprocess_output()

        head_image = image[coords[0][1]:coords[0][3], coords[0][0]:coords[0][2]]
        self.draw_outputs(coords, image)
        return coords, image, head_image

    def preprocess_input(self, image):
        input_dict = {}
        n, c, h, w = self.input_shape
        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        input_dict[self.input_name] = in_frame
        return input_dict

    def preprocess_output(self):
        res = self.exec_net.requests[0].outputs[self.output_blob]
        coords = []
        for obj in res[0][0]:
            if obj[2] > 0.6:  # args.threshold:
                xmin = int(obj[3] * self.initial_w)
                ymin = int(obj[4] * self.initial_h)
                xmax = int(obj[5] * self.initial_w)
                ymax = int(obj[6] * self.initial_h)
                coords.append((xmin, ymin, xmax, ymax))
        return coords

    def draw_outputs(self, coords, image):
        color = (10, 245, 10)
        for obj in coords:
            cv2.rectangle(image, (obj[0], obj[1]), (obj[2], obj[3]), color, 2)

    def set_initial(self, w, h):
        self.initial_w = w
        self.initial_h = h
