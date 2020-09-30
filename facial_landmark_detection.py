import cv2
import time


class LandmarkDetection:
    def __init__(self, model_name):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'

    def load_model(self, ie):
        self.net2 = ie.read_network(model=self.model_structure, weights=self.model_weights)
        self.exec_net = ie.load_network(network=self.net2, num_requests=0, device_name="MYRIAD")
        self.input_name = next(iter(self.exec_net.inputs))
        self.input_shape = self.exec_net.inputs[self.input_name].shape
        self.output_name = next(iter(self.exec_net.outputs))
        self.output_shape = self.exec_net.outputs[self.output_name].shape

    def predict(self, image):
        w = image.shape[1]
        h = image.shape[0]

        feed_dict = self.preprocess_input(image)
        self.exec_net.start_async(request_id=0, inputs=feed_dict)
        while True:
            status = self.exec_net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)

        coords = self.preprocess_output()
        self.draw_outputs(coords, image)

        left_eye = image[int(coords[1] * h) - 30:int(coords[1] * h) + 30,
                   int(coords[0] * w) - 30:int(coords[0] * w) + 30]
        right_eye = image[int(coords[3] * h) - 30:int(coords[3] * h) + 30,
                    int(coords[2] * w) - 30:int(coords[2] * w) + 30]
        nose = (coords[4][0], coords[5][0])

        return image, left_eye, right_eye, nose

    def preprocess_input(self, image):
        input_dict = {}
        n, c, h, w = self.input_shape
        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        input_dict[self.input_name] = in_frame
        return input_dict

    def preprocess_output(self):
        r_eye_x = self.exec_net.requests[0].outputs[self.output_name][0][0]
        r_eye_y = self.exec_net.requests[0].outputs[self.output_name][0][1]
        l_eye_x = self.exec_net.requests[0].outputs[self.output_name][0][2]
        l_eye_y = self.exec_net.requests[0].outputs[self.output_name][0][3]
        nose_x = self.exec_net.requests[0].outputs[self.output_name][0][4]
        nose_y = self.exec_net.requests[0].outputs[self.output_name][0][5]
        return (l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y)

    def draw_outputs(self, coords, image):
        w = image.shape[1]
        h = image.shape[0]

        eye_right_x, eye_right_y, left_eye_x, left_eye_y, nose_x, nose_y = coords
        color = (245, 245, 245)

        cv2.circle(image, (int(nose_x * w), int(nose_y * h)), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)

        cv2.rectangle(image, (int(eye_right_x * w) - 30, int(eye_right_y * h) - 30),
                      (int(eye_right_x * w) + 30, int(eye_right_y * h) + 30), color, 2)
        cv2.rectangle(image, (int(left_eye_x * w) - 30, int(left_eye_y * h) - 30),
                      (int(left_eye_x * w) + 30, int(left_eye_y * h) + 30), color, 2)
