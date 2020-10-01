import time
from pathlib import Path

from openvino.inference_engine import IENetwork, IECore
import cv2
from utils import draw_3d_axis


class HeadPose:
    def __init__(self, ie):
        self.model_weights = Path("models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.bin").resolve().absolute()
        self.model_structure = self.model_weights.with_suffix('.xml')
        self.model = IENetwork(str(self.model_structure), str(self.model_weights))
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self.net3 = ie.read_network(model=str(self.model_structure), weights=str(self.model_weights))
        self.exec_net = ie.load_network(network=self.net3, num_requests=0, device_name="MYRIAD")
        self.input_blob = next(iter(self.exec_net.inputs))
        self.output_blob = next(iter(self.exec_net.outputs))

    def predict(self, image, origin):
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

        head_pose = (
            self.exec_net.requests[0].outputs['angle_r_fc'],
            self.exec_net.requests[0].outputs['angle_p_fc'],
            self.exec_net.requests[0].outputs['angle_y_fc']
        )

        draw_3d_axis(image, head_pose[2], head_pose[1], head_pose[0], int(origin[0] * width), int(origin[1] * height))
        return head_pose, image

