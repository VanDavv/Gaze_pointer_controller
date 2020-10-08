import time
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
# from openvino.inference_engine import IENetwork, IECore
from utils import draw_3d_axis
import depthai

debug = True


def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True


def valid(*args):
    for arg in args:
        if arg is None:
            return False
        if isinstance(arg, np.ndarray) and arg.size == 0:
            return False

    return True


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def to_nn_result(raw_data: list):
    return np.frombuffer(bytes(raw_data), dtype=np.float16)


def to_bbox_result(raw_data: list):
    arr = to_nn_result(raw_data)
    arr = arr[:np.where(arr == -1)[0][0]]
    arr = arr.reshape((arr.size // 7, 7))
    return arr


class Main:
    def __init__(self):
        print("Loading input...")
        self.cap = cv2.VideoCapture(str(Path("demo.mp4").resolve().absolute()))
        self.create_pipeline()
        self.start_pipeline()
        # self.ie = IECore()
        # print("Loading networks...")
        # self.face_net = load_net(self.ie, Path("models/face-detection-retail-0004"))
        # self.landmark_net = load_net(self.ie, Path("models/landmarks-regression-retail-0009"))
        # self.pose_net = load_net(self.ie, Path("models/head-pose-estimation-adas-0001"))
        # self.gaze_net = load_net(self.ie, Path("models/gaze-estimation-adas-0002"))

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()
        
        # ColorCamera
        # print("Creating Color Camera...")
        # cam = self.pipeline.createColorCamera()
        # cam.setPreviewSize(300, 300)
        # cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        # cam.setInterleaved(False)
        # cam.setCamId(0)
        # cam_xout = self.pipeline.createXLinkOut()
        # cam_xout.setStreamName("preview")
        # cam.preview.link(cam_xout.input)

        # FrameInput
        frame_in = self.pipeline.createXLinkIn()
        frame_in.setStreamName("frame_in")

        
        # NeuralNetwork
        print("Creating Face Detection Neural Network...")
        face_nn = self.pipeline.createNeuralNetwork()
        face_nn.setBlobPath(str(Path("models/face-detection-retail-0004/face-detection-retail-0004.blob").resolve().absolute()))
        face_nn_xout = self.pipeline.createXLinkOut()
        face_nn_xout.setStreamName("face_nn")
        frame_in.out.link(face_nn.input)
        face_nn.out.link(face_nn_xout.input)
        
        # NeuralNetwork
        print("Creating Landmarks Detection Neural Network...")
        land_nn = self.pipeline.createNeuralNetwork()
        land_nn.setBlobPath(
            str(Path("models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.blob").resolve().absolute())
        )
        land_nn_xin = self.pipeline.createXLinkIn()
        land_nn_xin.setStreamName("landmark_in")
        land_nn_xin.out.link(land_nn.input)
        land_nn_xout = self.pipeline.createXLinkOut()
        land_nn_xout.setStreamName("landmark_nn")
        land_nn.out.link(land_nn_xout.input)

        # NeuralNetwork
        print("Creating Head Pose Neural Network...")
        pose_nn = self.pipeline.createNeuralNetwork()
        pose_nn.setBlobPath(
            str(Path("models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.blob").resolve().absolute())
        )
        pose_nn_xin = self.pipeline.createXLinkIn()
        pose_nn_xin.setStreamName("pose_in")
        pose_nn_xin.out.link(pose_nn.input)
        pose_nn_xout = self.pipeline.createXLinkOut()
        pose_nn_xout.setStreamName("pose_nn")
        pose_nn.out.link(pose_nn_xout.input)

        print("Pipeline created.")

    def start_pipeline(self):
        found, deviceInfo = depthai.XLinkConnection.getFirstDevice(depthai.X_LINK_UNBOOTED)
        if not found:
            raise RuntimeError("Device not found")
        print("Device found.")
        self.device = depthai.Device(deviceInfo)
        print("Starting pipeline...")
        self.device.startPipeline(self.pipeline)
        self.frame_in = self.device.getInputQueue("frame_in")
        self.face_nn = self.device.getOutputQueue("face_nn")
        self.land_in = self.device.getInputQueue("landmark_in")
        self.land_nn = self.device.getOutputQueue("landmark_nn")
        self.pose_in = self.device.getInputQueue("pose_in")
        self.pose_nn = self.device.getOutputQueue("pose_nn")

    def run_face(self, frame):
        buff = depthai.RawBuffer()
        buff.data = to_planar(frame, (300, 300))
        self.frame_in.send(buff)
        has_results = wait_for_results(self.face_nn)
        if not has_results:
            print("No data from face_nn, skipping frame...")
            return None
        results = to_bbox_result(self.face_nn.get().data)
        height, width = frame.shape[:2]
        coords = [
            (int(obj[3] * width), int(obj[4] * height), int(obj[5] * width), int(obj[6] * height))
            for obj in results
            if obj[2] > 0.4
        ]
        if len(coords) == 0:
            return None
        head_image = frame[coords[0][1]:coords[0][3], coords[0][0]:coords[0][2]]
        if debug:
            for obj in coords:
                cv2.rectangle(frame, (obj[0], obj[1]), (obj[2], obj[3]), (10, 245, 10), 2)
        return head_image

    def run_landmark(self, face_frame):
        buff = depthai.RawBuffer()
        buff.data = to_planar(face_frame, (48, 48))
        self.land_in.send(buff)
        has_results = wait_for_results(self.land_nn)
        if not has_results:
            print("No data from land_nn, skipping frame...")
            return None, None, None

        out = to_nn_result(self.land_nn.get().data)
        right_eye, left_eye, nose = out[:2], out[2:4], out[4:6]
        h, w = face_frame.shape[:2]

        right_eye_image = face_frame[int(right_eye[1]*h) - 30:int(right_eye[1]*h) + 30, int(right_eye[0]*w) - 30:int(right_eye[0]*w) + 30]
        left_eye_image = face_frame[int(left_eye[1]*h) - 30:int(left_eye[1]*h) + 30, int(left_eye[0]*w) - 30:int(left_eye[0]*w) + 30]

        if debug:
            cv2.circle(face_frame, (int(nose[0] * w), int(nose[1] * h)), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)
            cv2.rectangle(face_frame, (int(right_eye[0] * w - 30), int(right_eye[1] * h - 30)), (int(right_eye[0] * w + 30), int(right_eye[1] * h + 30)), 245, 2)
            cv2.rectangle(face_frame, (int(left_eye[0] * w - 30), int(left_eye[1] * h - 30)), (int(left_eye[0] * w + 30), int(left_eye[1] * h + 30)), 245, 2)

        return left_eye_image, right_eye_image, nose

    def run_pose(self, face_frame, nose):
        buff = depthai.RawBuffer()
        buff.data = to_planar(face_frame, (60, 60))
        self.pose_in.send(buff)
        has_results = wait_for_results(self.pose_nn)
        if not has_results:
            print("No data from pose_nn, skipping frame...")
            return None

        raw = self.pose_nn.get()
        out = to_nn_result(raw.data)
        print("RAW", raw.data)
        print("OUT", out)
        return None
        # head_pose = [value[0] for value in out.values()]
        #
        # if debug:
        #     height, width = face_frame.shape[:2]
        #     draw_3d_axis(face_frame, head_pose[2], head_pose[1], head_pose[0], int(nose[0] * width), int(nose[1] * height))
        # return head_pose

    def run_gaze(self, l_eye, r_eye, pose):
        out = run_net(self.gaze_net, {
            "left_eye_image": l_eye,
            "right_eye_image": r_eye,
            "head_pose_angles": pose
        })

        if debug:
            origin_x_re = r_eye.shape[1] // 2
            origin_y_re = r_eye.shape[0] // 2
            origin_x_le = l_eye.shape[1] // 2
            origin_y_le = l_eye.shape[0] // 2

            x, y = (out["gaze_vector"] * 100).astype(int)[:2]
            cv2.arrowedLine(l_eye, (origin_x_le, origin_y_le), (origin_x_le + x, origin_y_le - y), (255, 0, 255), 3)
            cv2.arrowedLine(r_eye, (origin_x_re, origin_y_re), (origin_x_re + x, origin_y_re - y), (255, 0, 255), 3)
        return out["gaze_vector"]

    def parse(self, frame):
        face_image = self.run_face(frame)
        if valid(face_image):
            cv2.imshow("face", face_image)
            left_eye, right_eye, nose = self.run_landmark(face_image)
            if valid(left_eye, right_eye, nose):
                cv2.imshow("left_eye", left_eye)
                cv2.imshow("right_eye", right_eye)

                pose = self.run_pose(face_image, nose)
        # if left_eye.size > 0 and right_eye.size > 0:
        #     eye_pose = self.run_gaze(left_eye, right_eye, pose)
        #     print(eye_pose)

    def run(self):
        while self.cap.isOpened():
            read_correctly, frame = self.cap.read()
            if not read_correctly:
                break

            self.parse(frame)

            if debug:
                cv2.imshow("Camera_view", cv2.resize(frame, (900, 450)))
                if cv2.waitKey(1) == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Main().run()
