import cv2
import numpy as np
import time
import math
import mediapipe as mp
# from some_pose_detection_module import poseDetector

class poseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,  # মডেল কমপ্লেক্সিটি সঠিকভাবে সেট করা হয়েছে
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

class ExerciseDetector(poseDetector):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.stage = None
        self.curl_count = 0
        self.dir = 0

    def detectPushUp(self, img):
        angle = self.findAngle(img, 11, 13, 15)  # Shoulder, elbow, wrist
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
        return self.counter

    def detectSitUp(self, img):
        angle = self.findAngle(img, 11, 23, 25)  # Shoulder, hip, knee
        if angle > 140:
            self.stage = "down"
        if angle < 100 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        return self.counter

    def detectJumping(self, img):
        angle = self.findAngle(img, 23, 25, 27)  # Hip, knee, ankle
        if angle > 160:
            self.stage = "up"
        if angle < 110 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
        return self.counter

    def detectCurl(self, img):
        angle = self.findAngle(img, 11, 13, 15)  # Shoulder, elbow, wrist
        if angle > 160:
            self.dir = 0
        elif angle < 60 and self.dir == 0:
            self.dir = 1
            self.curl_count += 1
        return self.curl_count

    def detectOnSpotRunning(self, img):
        left_foot = self.lmList[31][2]  # Left ankle y-coordinate
        right_foot = self.lmList[32][2]  # Right ankle y-coordinate
        if abs(left_foot - right_foot) > 50:
            self.stage = "running"
        return "Running" if self.stage == "running" else "Not running"

    def detectPlank(self, img):
        angle = self.findAngle(img, 11, 23, 25)
        if 170 < angle < 180:
            self.stage = "straight"
        return "Plank detected!" if self.stage == "straight" else "Not a plank"

    def detectRussianTwist(self, img):
        angle = self.findAngle(img, 11, 23, 12)  # Left shoulder, hip, right shoulder
        if angle < 90:
            self.stage = "left"
        elif angle > 90:
            self.stage = "right"
        return self.stage

    def detectSquats(self, img):
        angle = self.findAngle(img, 23, 25, 27)  # Hip, knee, ankle
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
        return self.counter

    def detectDips(self, img):
        angle = self.findAngle(img, 11, 13, 15)  # Shoulder, elbow, wrist
        if angle > 90:
            self.stage = "up"
        if angle < 45 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
        return self.counter

    def detectLunges(self, img):
        angle = self.findAngle(img, 23, 25, 27)  # Hip, knee, ankle
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
        return self.counter

    def detectFlutterKicks(self, img):
        left_foot = self.lmList[31][2]  # Left ankle y-coordinate
        right_foot = self.lmList[32][2]  # Right ankle y-coordinate
        if abs(left_foot - right_foot) > 50:
            self.stage = "flutter"
            self.counter += 1
        return self.counter


# Main function
def main():
    cap = cv2.VideoCapture(0)  # Use webcam
    pTime = 0
    detector = ExerciseDetector()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            # Detect exercises
            push_up_count = detector.detectPushUp(img)
            sit_up_count = detector.detectSitUp(img)
            jump_count = detector.detectJumping(img)
            curl_count = detector.detectCurl(img)
            running_status = detector.detectOnSpotRunning(img)
            plank_status = detector.detectPlank(img)
            twist_status = detector.detectRussianTwist(img)
            squat_count = detector.detectSquats(img)
            dip_count = detector.detectDips(img)
            lunge_count = detector.detectLunges(img)
            flutter_count = detector.detectFlutterKicks(img)

            # Display results
            cv2.putText(img, f'Push-ups: {push_up_count}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Sit-ups: {sit_up_count}', (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Jumping: {jump_count}', (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Curls: {curl_count}', (50, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Running: {running_status}', (50, 300), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Plank: {plank_status}', (50, 350), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Twist: {twist_status}', (50, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Squats: {squat_count}', (50, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Dips: {dip_count}', (50, 500), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Lunges: {lunge_count}', (50, 550), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, f'Flutter Kicks: {flutter_count}', (50, 600), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS
        cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # Show Image
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()