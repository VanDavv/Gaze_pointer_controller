from pathlib import Path

import cv2

import logging as log
from openvino.inference_engine import IECore
from face_detection import FaceDetection
from head_pose_estimation import HeadPose
from facial_landmark_detection import LandmarkDetection
from gaze_estimation import GazeEstimation


def main():
    video_file = str(Path("demo.mp4").resolve().absolute())

    log.info("Creating fd Inference Engine...")
    ie = IECore()

    fd = FaceDetection()
    hp = HeadPose()
    ld = LandmarkDetection()
    ge = GazeEstimation()

    fd.load_model(ie)
    hp.load_model(ie)
    ld.load_model(ie)
    ge.load_model(ie)

    try:
        cap = cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: " + video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    counter = 0

    try:
        fd.set_initial(initial_w, initial_h)
        hp.set_initial(initial_w, initial_h)
        ge.set_initial(initial_w, initial_h)

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
    except Exception as e:
        print("Could not run Inference: ", e)


if __name__ == '__main__':
    main()
    exit(0)
