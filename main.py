import time
from pathlib import Path
import cv2
from openvino.inference_engine import IENetwork, IECore
from utils import draw_3d_axis


class FaceDetection:
    def __init__(self, ie):
        self.model_weights = Path("models/face-detection-retail-0004/face-detection-retail-0004.bin").resolve().absolute()
        self.model_structure = self.model_weights.with_suffix('.xml')

        self.model = IENetwork(str(self.model_structure), str(self.model_weights))

        self.input_shape = self.model.inputs["data"].shape
        self.output_shape = self.model.outputs["detection_out"].shape
        self.net = ie.read_network(model=str(self.model_structure), weights=str(self.model_weights))
        self.exec_net = ie.load_network(network=self.net, num_requests=0, device_name="MYRIAD")

    def predict(self, image):
        height, width = image.shape[:2]
        input_dict = {}
        n, c, h, w = self.input_shape
        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        input_dict["data"] = in_frame
        self.exec_net.start_async(request_id=0, inputs=input_dict)
        while True:
            status = self.exec_net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(0.1)
        coords = []
        for obj in self.exec_net.requests[0].outputs["detection_out"][0][0]:
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


class LandmarkDetection:
    def __init__(self, ie):
        self.model_weights = Path("models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.bin").resolve().absolute()
        self.model_structure = self.model_weights.with_suffix('.xml')
        self.net2 = ie.read_network(model=str(self.model_structure), weights=str(self.model_weights))
        self.exec_net = ie.load_network(network=self.net2, num_requests=0, device_name="MYRIAD")
        self.input_shape = self.exec_net.inputs["0"].shape
        self.output_shape = self.exec_net.outputs["95"].shape

    def predict(self, image):
        input_dict = {}
        n, c, h, w = self.input_shape
        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        input_dict["0"] = in_frame
        self.exec_net.start_async(request_id=0, inputs=input_dict)
        while True:
            status = self.exec_net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(0.1)
        right_eye = (
             self.exec_net.requests[0].outputs["95"][0][0],
             self.exec_net.requests[0].outputs["95"][0][1],
        )
        left_eye = (
             self.exec_net.requests[0].outputs["95"][0][2],
             self.exec_net.requests[0].outputs["95"][0][3],
        )
        nose = (
             self.exec_net.requests[0].outputs["95"][0][4],
             self.exec_net.requests[0].outputs["95"][0][5],
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


class HeadPose:
    def __init__(self, ie):
        self.model_weights = Path("models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.bin").resolve().absolute()
        self.model_structure = self.model_weights.with_suffix('.xml')
        self.model = IENetwork(str(self.model_structure), str(self.model_weights))
        self.input_shape = self.model.inputs["data"].shape
        self.output_shape = self.model.outputs["angle_p_fc"].shape
        self.net3 = ie.read_network(model=str(self.model_structure), weights=str(self.model_weights))
        self.exec_net = ie.load_network(network=self.net3, num_requests=0, device_name="MYRIAD")

    def predict(self, image, origin):
        height, width = image.shape[:2]
        input_dict = {}
        n, c, h, w = self.input_shape
        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        input_dict["data"] = in_frame
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


class GazeEstimation:
    def __init__(self, ie):
        self.model_weights = Path("models/gaze-estimation-adas-0002/gaze-estimation-adas-0002.bin").resolve().absolute()
        self.model_structure = self.model_weights.with_suffix('.xml')
        self.net4 = ie.read_network(model=str(self.model_structure), weights=str(self.model_weights))
        self.exec_net = ie.load_network(network=self.net4, num_requests=0, device_name="MYRIAD")
        self.input_shape = self.exec_net.inputs['left_eye_image'].shape

    def predict(self, r_eye, l_eye, pose):
        input_dict = {}
        input_dict['left_eye_image'] = self.preprocess_input(r_eye)
        input_dict['right_eye_image'] = self.preprocess_input(l_eye)
        input_dict['head_pose_angles'] = [pose]

        self.exec_net.start_async(request_id=0, inputs=input_dict)
        while True:
            status = self.exec_net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
        coords = (
            self.exec_net.requests[0].outputs['gaze_vector'][0][0],
            self.exec_net.requests[0].outputs['gaze_vector'][0][1],
            self.exec_net.requests[0].outputs['gaze_vector'][0][2],
        )
        origin_x_re = r_eye.shape[1] // 2
        origin_y_re = r_eye.shape[0] // 2
        origin_x_le = l_eye.shape[1] // 2
        origin_y_le = l_eye.shape[0] // 2

        x, y = int(coords[0] * 100), int(coords[1] * 100)
        cv2.arrowedLine(l_eye, (origin_x_le, origin_y_le), (origin_x_le + x, origin_y_le - y), (255, 0, 255), 3)
        cv2.arrowedLine(r_eye, (origin_x_re, origin_y_re), (origin_x_re + x, origin_y_re - y), (255, 0, 255), 3)
        return coords

    def preprocess_input(self, image):
        try:
            n, c, h, w = self.input_shape
            in_frame = cv2.resize(image, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            return in_frame
        except Exception as e:
            print(str(e))


if __name__ == '__main__':
    video_file = str(Path("demo.mp4").resolve().absolute())

    cap = cv2.VideoCapture(video_file)

    ie = IECore()

    fd = FaceDetection(ie)
    hp = HeadPose(ie)
    ld = LandmarkDetection(ie)
    ge = GazeEstimation(ie)

    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        coords, image, head_image = fd.predict(frame)

        land_image, left_eye, right_eye, nose = ld.predict(head_image)
        head_pose, pose_image = hp.predict(head_image, nose)
        eye_pose = ge.predict(left_eye, right_eye, head_pose)
        print(eye_pose)
        cv2.imshow("Camera_view", cv2.resize(image, (900, 450)))

    cap.release()
    cv2.destroyAllWindows()
