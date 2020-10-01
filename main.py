import time
from pathlib import Path
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
from utils import draw_3d_axis


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


def load_net(ie, dir: Path):
    definition = str(next(dir.glob("*.xml")).resolve().absolute())
    weights = str(next(dir.glob("*.bin")).resolve().absolute())
    net = ie.read_network(model=definition, weights=weights)
    return ie.load_network(network=net, num_requests=0, device_name="MYRIAD")


class FaceDetection:
    def __init__(self, ie):
        self.net = load_net(ie, Path("models/face-detection-retail-0004"))

    def predict(self, image):
        out = run_net(self.net, {"data": image})
        height, width = image.shape[:2]
        coords = [
            (int(obj[3] * width), int(obj[4] * height), int(obj[5] * width), int(obj[6] * height))
            for obj in out["detection_out"][0]
            if obj[2] > 0.6
        ]
        head_image = image[coords[0][1]:coords[0][3], coords[0][0]:coords[0][2]]
        for obj in coords:
            cv2.rectangle(image, (obj[0], obj[1]), (obj[2], obj[3]), (10, 245, 10), 2)
        return head_image


class LandmarkDetection:
    def __init__(self, ie):
        self.net = load_net(ie, Path("models/landmarks-regression-retail-0009"))

    def predict(self, image):
        out = run_net(self.net, {"0": image})
        right_eye, left_eye, nose = out["95"][:2], out["95"][2:4], out["95"][4:]
        w = image.shape[1]
        h = image.shape[0]

        right_eye_image = image[int(right_eye[1]*h) - 30:int(right_eye[1]*h) + 30, int(right_eye[0]*w) - 30:int(right_eye[0]*w) + 30]
        left_eye_image = image[int(left_eye[1]*h) - 30:int(left_eye[1]*h) + 30, int(left_eye[0]*w) - 30:int(left_eye[0]*w) + 30]

        cv2.circle(image, (int(nose[0] * w), int(nose[1] * h)), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)
        cv2.rectangle(image, (right_eye[0] * w - 30, right_eye[1] * h - 30), (right_eye[0] * w + 30, right_eye[1] * h + 30), (245, 245, 245), 2)
        cv2.rectangle(image, (left_eye[0] * w - 30, left_eye[1] * h - 30), (left_eye[0] * w + 30, left_eye[1] * h + 30), (245, 245, 245), 2)

        return left_eye_image, right_eye_image, nose


class HeadPose:
    def __init__(self, ie):
        self.net = load_net(ie, Path("models/head-pose-estimation-adas-0001"))

    def predict(self, image, origin):
        out = run_net(self.net, {"data": image})
        head_pose = [value[0] for value in out.values()]
        height, width = image.shape[:2]
        draw_3d_axis(image, head_pose[2], head_pose[1], head_pose[0], int(origin[0] * width), int(origin[1] * height))
        return head_pose


class GazeEstimation:
    def __init__(self, ie):
        self.net = load_net(ie, Path("models/gaze-estimation-adas-0002"))

    def predict(self, l_eye, r_eye, pose):
        out = run_net(self.net, {
            "left_eye_image": l_eye,
            "right_eye_image": r_eye,
            "head_pose_angles": pose
        })

        origin_x_re = r_eye.shape[1] // 2
        origin_y_re = r_eye.shape[0] // 2
        origin_x_le = l_eye.shape[1] // 2
        origin_y_le = l_eye.shape[0] // 2

        x, y = (out["gaze_vector"] * 100).astype(int)[:2]
        cv2.arrowedLine(l_eye, (origin_x_le, origin_y_le), (origin_x_le + x, origin_y_le - y), (255, 0, 255), 3)
        cv2.arrowedLine(r_eye, (origin_x_re, origin_y_re), (origin_x_re + x, origin_y_re - y), (255, 0, 255), 3)
        return out["gaze_vector"]


if __name__ == '__main__':
    cap = cv2.VideoCapture(str(Path("demo.mp4").resolve().absolute()))

    ie = IECore()

    fd = FaceDetection(ie)
    hp = HeadPose(ie)
    ld = LandmarkDetection(ie)
    ge = GazeEstimation(ie)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) == ord('q'):
            break

        head_image = fd.predict(frame)
        left_eye, right_eye, nose = ld.predict(head_image)
        head_pose = hp.predict(head_image, nose)
        if left_eye.size > 0 and right_eye.size > 0:
            eye_pose = ge.predict(left_eye, right_eye, head_pose)
            print(eye_pose)
        cv2.imshow("Camera_view", cv2.resize(frame, (900, 450)))

    cap.release()
    cv2.destroyAllWindows()
