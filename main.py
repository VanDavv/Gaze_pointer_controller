from pathlib import Path

import cv2

from openvino.inference_engine import IECore
from face_detection import FaceDetection
from head_pose_estimation import HeadPose
from facial_landmark_detection import LandmarkDetection
from gaze_estimation import GazeEstimation


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
