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


def to_nn_result(nn_data):
    return np.array(nn_data.getFirstLayerFp16())


def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


def to_bbox_result(nn_data):
    arr = to_nn_result(nn_data)
    arr = arr[:np.where(arr == -1)[0][0]]
    arr = arr.reshape((arr.size // 7, 7))
    return arr


def to_bytes_aligned(data: bytes):
    return data + np.zeros(64 - len(data) % 64, dtype=np.float16).tobytes()


class Main:
    def __init__(self):
        print("Loading input...")
        self.cap = cv2.VideoCapture(str(Path("demo.mp4").resolve().absolute()))
        self.create_pipeline()
        self.start_pipeline()

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

        # NeuralNetwork
        print("Creating Gaze Estimation Neural Network...")
        gaze_nn = self.pipeline.createNeuralNetwork()
        gaze_nn.setBlobPath(
            str(Path("models/gaze-estimation-adas-0002/gaze-estimation-adas-0002.blob").resolve().absolute())
        )
        gaze_nn_xin = self.pipeline.createXLinkIn()
        gaze_nn_xin.setStreamName("gaze_in")
        gaze_nn_xin.out.link(gaze_nn.input)
        gaze_nn_xout = self.pipeline.createXLinkOut()
        gaze_nn_xout.setStreamName("gaze_nn")
        gaze_nn.out.link(gaze_nn_xout.input)

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
        self.gaze_in = self.device.getInputQueue("gaze_in")
        self.gaze_nn = self.device.getOutputQueue("gaze_nn")

    def run_face(self, frame):
        nn_data = depthai.NNData()
        nn_data.setLayer("data", to_planar(frame, (300, 300)))
        self.frame_in.send(nn_data)
        has_results = wait_for_results(self.face_nn)
        if not has_results:
            print("No data from face_nn, skipping frame...")
            return None
        results = to_bbox_result(self.face_nn.get())
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
        nn_data = depthai.NNData()
        nn_data.setLayer("0", to_planar(face_frame, (48, 48)))
        self.land_in.send(nn_data)
        has_results = wait_for_results(self.land_nn)
        if not has_results:
            print("No data from land_nn, skipping frame...")
            return None, None, None

        out = to_nn_result(self.land_nn.get())
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
        nn_data = depthai.NNData()
        nn_data.setLayer("data", to_planar(face_frame, (60, 60)))
        self.pose_in.send(nn_data)
        has_results = wait_for_results(self.pose_nn)
        if not has_results:
            print("No data from pose_nn, skipping frame...")
            return None

        head_pose = [val[0] for val in to_tensor_result(self.pose_nn.get()).values()]

        if debug:
            height, width = face_frame.shape[:2]
            draw_3d_axis(face_frame, head_pose[2], head_pose[1], head_pose[0], int(nose[0] * width), int(nose[1] * height))
        return head_pose

    def run_gaze(self, l_eye: np.ndarray, r_eye: np.ndarray, pose):
        nn_data = depthai.NNData()
        nn_data.setLayer("lefy_eye_image", to_planar(l_eye, (60, 60)))
        nn_data.setLayer("right_eye_image", to_planar(r_eye, (60, 60)))
        nn_data.setLayer("head_pose_angles", pose)
        self.gaze_in.send(nn_data)
        has_results = wait_for_results(self.gaze_nn)
        if not has_results:
            print("No data from gaze_nn, skipping frame...")
            return None

        gaze = to_nn_result(self.gaze_nn.get())

        if debug:
            origin_x_re = r_eye.shape[1] // 2
            origin_y_re = r_eye.shape[0] // 2
            origin_x_le = l_eye.shape[1] // 2
            origin_y_le = l_eye.shape[0] // 2

            x, y = (gaze * 100).astype(int)[:2]
            cv2.arrowedLine(l_eye, (origin_x_le, origin_y_le), (origin_x_le + x, origin_y_le - y), (255, 0, 255), 3)
            cv2.arrowedLine(r_eye, (origin_x_re, origin_y_re), (origin_x_re + x, origin_y_re - y), (255, 0, 255), 3)
        return gaze

    def parse(self, frame):
        face_image = self.run_face(frame)
        if valid(face_image):
            left_eye, right_eye, nose = self.run_landmark(face_image)
            if valid(left_eye, right_eye, nose):
                pose = self.run_pose(face_image, nose)
                if valid(pose):
                    eye_pose = self.run_gaze(left_eye, right_eye, pose)
                    print(eye_pose)

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
        del self.device


if __name__ == '__main__':
    Main().run()
