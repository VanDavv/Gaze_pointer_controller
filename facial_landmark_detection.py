from pathlib import Path

import cv2
import time


class LandmarkDetection:
    def __init__(self, ie):
        self.model_weights = Path("models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.bin").resolve().absolute()
        self.model_structure = self.model_weights.with_suffix('.xml')
        self.net2 = ie.read_network(model=str(self.model_structure), weights=str(self.model_weights))
        self.exec_net = ie.load_network(network=self.net2, num_requests=0, device_name="MYRIAD")
        self.input_name = next(iter(self.exec_net.inputs))
        self.input_shape = self.exec_net.inputs[self.input_name].shape
        self.output_name = next(iter(self.exec_net.outputs))
        self.output_shape = self.exec_net.outputs[self.output_name].shape

    def predict(self, image):
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
        right_eye = (
             self.exec_net.requests[0].outputs[self.output_name][0][0],
             self.exec_net.requests[0].outputs[self.output_name][0][1],
        )
        left_eye = (
             self.exec_net.requests[0].outputs[self.output_name][0][2],
             self.exec_net.requests[0].outputs[self.output_name][0][3],
        )
        nose = (
             self.exec_net.requests[0].outputs[self.output_name][0][4],
             self.exec_net.requests[0].outputs[self.output_name][0][5],
        )
        w = image.shape[1]
        h = image.shape[0]

        right_eye_image = image[int(right_eye[1]*h) - 30:int(right_eye[1]*h) + 30, int(right_eye[0]*w) - 30:int(right_eye[0]*w) + 30]
        left_eye_image = image[int(left_eye[1]*h) - 30:int(left_eye[1]*h) + 30, int(left_eye[0]*w) - 30:int(left_eye[0]*w) + 30]

        cv2.circle(image, (int(nose[0] * w), int(nose[1] * h)), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)
        cv2.rectangle(image, (right_eye[0] * w - 30, right_eye[1] * h - 30),
                      (right_eye[0] * w + 30, right_eye[1] * h + 30), (245, 245, 245), 2)
        cv2.rectangle(image, (left_eye[0] * w - 30, left_eye[1] * h - 30),
                      (left_eye[0] * w + 30, left_eye[1] * h + 30), (245, 245, 245), 2)

        return image, left_eye_image, right_eye_image, nose


