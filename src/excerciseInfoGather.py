from excerciseType import excerciseType
import cv2
from poseDetector import poseDetector
cap = cv2.VideoCapture('C:/Users/Checkout/Downloads/test/s02/videos/squat.mp4')
detector = poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    left_leg_angle = excerciseType(detector.results.pose_landmarks.landmark).angle_of_the_right_leg()
    right_leg_angle = excerciseType(detector.results.pose_landmarks.landmark).angle_of_the_left_leg()
    avg_leg_angle = (left_leg_angle + right_leg_angle) // 2
    cv2.putText(img, "Angle : " + str(avg_leg_angle), (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()