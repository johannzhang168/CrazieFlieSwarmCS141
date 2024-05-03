import os
import sys
import subprocess
import importlib.util

import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import KeyPointClassifier
from model import PointHistoryClassifier

from crazyflie_py import Crazyswarm

swarm = Crazyswarm()
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs

def takeoff(target_height=1.0, duration=1.0):
    """ Command all drones to take off to a specified height and update initialPosition """
    print("Taking off to height:", target_height)
    canTakeOff = True
    for cf in allcfs.crazyflies:
        if cf.initialPosition[2] != 0:
            canTakeOff = False
    if canTakeOff:
        allcfs.takeoff(targetHeight=target_height, duration=duration)
        timeHelper.sleep(duration + 0.5)  # Wait for the takeoff to complete
        for cf in allcfs.crazyflies:
            # Update the initial position with the new height, keeping X and Y the same
            initial_x, initial_y, _ = cf.initialPosition
            cf.initialPosition = np.array([initial_x, initial_y, target_height])

def move(cf, delta):
    """ Move function to handle position changes """
    new_position = np.array(cf.initialPosition) + np.array(delta)
    cf.goTo(new_position, 0, 1.0)
    cf.initialPosition = new_position  # Update drone's position after the move

def go_up():
    for cf in allcfs.crazyflies:
        move(cf, [0, 0, 0.2])  # Change only Z-axis

def go_down():
    for cf in allcfs.crazyflies:
        move(cf, [0, 0, -0.2])  # Change only Z-axis

def go_left():
    for cf in allcfs.crazyflies:
        move(cf, [0, 0.2, 0])  # Change only Y-axis

def go_right():
    for cf in allcfs.crazyflies:
        move(cf, [0, -0.2, 0])  # Change only Y-axis

def go_fw():
    for cf in allcfs.crazyflies:
        move(cf, [0.2, 0, 0])  # Change only X-axis

def go_bk():
    for cf in allcfs.crazyflies:
        move(cf, [-0.2, 0, 0])  # Change only X-axis

def land():
    allcfs.land(targetHeight=0.02, duration=1.0)



# Load the emotion detection model
face_classifier = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Setup drone swarm
swarm = Crazyswarm()
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs

def detect_emotion(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            return label
    return None


def reset_to_default():
    """ Reset each drone's initialPosition to ensure consistency with stored default settings """
    print("Confirming default initial positions for all drones...")
    for cf in allcfs.crazyflies:
        # Simply reaffirming the initialPosition to itself (logically redundant but clarifies intent)
        default_position = cf.initialPosition  # Assuming this fetches the stored YAML value correctly
        cf.initialPosition = np.array(default_position)  # Resetting it back to ensure no accidental modifications
        print(f"Drone's initial position reaffirmed to: {cf.initialPosition}")


def get_landing_positions():
    """ Returns the landing positions for all drones, maintaining X and Y but setting Z to 0 """
    landing_positions = []
    for cf in allcfs.crazyflies:
        current_position = np.array(cf.initialPosition)  # Get the current position
        landing_position = np.copy(current_position)
        landing_position[2] = 0  # Set Z-axis to 0
        landing_positions.append(landing_position)
        print(f"Drone landing at position: {landing_position}")
    return landing_positions






# sys.path.append('/home/ilikerustoo/ros2_ws/src/crazyswarm2/crazyflie_examples/crazyflie_examples')
# import nice_hover

# def import_from_path(name, path):
#     spec = importlib.util.spec_from_file_location(name, path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module

# nice_hover = import_from_path("nice_hover", '/home/ilikerustoo/ros2_ws/src/crazyswarm2/crazyflie_examples/crazyflie_examples/figure8.py')


def run_nice_hover():

    # script_path = '/home/ilikerustoo/ros2_ws/src/crazyswarm2/crazyflie_examples/crazyflie_examples'

    # result = subprocess.run(['python3', script_path], capture_output=True, text=True)

    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)


    # nice_hover.main()

    command = [

    'ros2', 'run', 'crazyflie_examples', 'figure8', '--ros-args', '-p', 'use_sim_time:=True'
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("command success")
        print("Output:", result.stdout)
    else:
        print("Error in executing command")
        print("Error:", result.stderr)
        



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    while True:
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Emotion detection
        emotion_label = detect_emotion(image)
        print(f"Detected emotion: {emotion_label}")
        # Here you can add if statements to react based on `emotion_label`
        # Example:
        # if emotion_label == 'Happy':
        #     go_up()
        # elif emotion_label == 'Sad':
        #     go_down()

        # Hand gesture detection
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Drawing hand landmarks
                debug_image = draw_landmarks(debug_image, hand_landmarks)

                # Hand sign classification
                hand_sign_id = keypoint_classifier.calculate(debug_image, hand_landmarks)
                print(f"Detected hand sign: {hand_sign_id}")
                # Perform actions based on hand sign detected
                # Placeholder for action commands based on hand sign and handedness
                current_time = time.time()
                last_time = last_recognized_time.get(hand_sign_id, 0)
                if current_time - last_time >= 2:   #timer to delay gesture sampling, adjust/experiment
                    last_recognized_time[hand_sign_id] = current_time
                    if hand_sign_id == 2:  # Point gesture
                        if (handedness.classification[0].label[0] == "L"):
                            print("GoBackwards")
                            # go_bk(allcfs)
                            go_bk()
                        else:
                            print("GoForwards")
                            # go_fw(allcfs)
                            go_fw()
                    elif hand_sign_id == 0:
                        if(handedness.classification[0].label[0] == "L"):
                            print("Land")
                            # land(allcfs)
                            land()
                            get_landing_positions()
                            print("Operations completed. Exiting now.")
                            sys.exit()
                            # reset_to_default()
                        else:
                    
                            print("TakeOff")
                            takeoff()
                    elif hand_sign_id == 1:
                        if(handedness.classification[0].label[0] == "L"):
                            print("GoLeft")
                            # go_left(allcfs)
                            go_left()
                        else:
                            print("GoRight")
                            # go_right(allcfs)
                            go_right()
                    elif hand_sign_id == 3:
                        if(handedness.classification[0].label[0] == "L"):
                            print("MoveDown")
                            # go_down(allcfs)
                            go_down()
                        else:
                            print("MoveUp")
                            # go_up(allcfs)
                            go_up()
            

# def main():
#     args = get_args()

#     cap_device = args.device
#     cap_width = args.width
#     cap_height = args.height

#     use_static_image_mode = args.use_static_image_mode
#     min_detection_confidence = args.min_detection_confidence
#     min_tracking_confidence = args.min_tracking_confidence

#     use_brect = True

#     cap = cv.VideoCapture(cap_device)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

#     # Model load #############################################################
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=use_static_image_mode,
#         max_num_hands=1,
#         min_detection_confidence=min_detection_confidence,
#         min_tracking_confidence=min_tracking_confidence,
#     )

#     keypoint_classifier = KeyPointClassifier()

#     # point_history_classifier = PointHistoryClassifier()

#     # Read labels ###########################################################
#     # with open('model/keypoint_classifier/keypoint_classifier_label.csv',
#     with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/keypoint_classifier/keypoint_classifier_label.csv'),
#               encoding='utf-8-sig') as f:
#         keypoint_classifier_labels = csv.reader(f)
#         keypoint_classifier_labels = [
#             row[0] for row in keypoint_classifier_labels
#         ]
#     # with open(
#     #         'model/point_history_classifier/point_history_classifier_label.csv',
#     #         encoding='utf-8-sig') as f:
#     #     point_history_classifier_labels = csv.reader(f)
#     #     point_history_classifier_labels = [
#     #         row[0] for row in point_history_classifier_labels
#     #     ]

#     # Coordinate history #################################################################
#     history_length = 16
#     point_history = deque(maxlen=history_length)

#     # Finger gesture history ################################################
#     finger_gesture_history = deque(maxlen=history_length)

#     #  ########################################################################
#     mode = 0
#     prev_hand_sign_id = ""
#     last_recognized_time = {}

#     while True:
#         # fps = cvFpsCalc.get()

#         # Process Key (ESC: end) #################################################
#         key = cv.waitKey(10)
#         if key == 27:  # ESC
#             break
#         number, mode = select_mode(key, mode)

#         # Camera capture #####################################################
#         ret, image = cap.read()
#         if not ret:
#             break
#         image = cv.flip(image, 1)  # Mirror display
#         emotion_label = detect_emotion(image)
#         print(f"Detected emotion: {emotion_label}")

#         debug_image = copy.deepcopy(image)

#         # Detection implementation #############################################################
#         image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

#         image.flags.writeable = False
#         results = hands.process(image)
#         image.flags.writeable = True

#         #  ####################################################################
#         if results.multi_hand_landmarks is not None:
#             for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
#                                                   results.multi_handedness):
#                 # Bounding box calculation
#                 brect = calc_bounding_rect(debug_image, hand_landmarks)
#                 # Landmark calculation
#                 landmark_list = calc_landmark_list(debug_image, hand_landmarks)

#                 # Conversion to relative coordinates / normalized coordinates
#                 pre_processed_landmark_list = pre_process_landmark(
#                     landmark_list)
#                 pre_processed_point_history_list = pre_process_point_history(
#                     debug_image, point_history)
#                 # Write to the dataset file
#                 logging_csv(number, mode, pre_processed_landmark_list,
#                             pre_processed_point_history_list)

#                 # Hand sign classification
#                 hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

#                 current_time = time.time()
#                 last_time = last_recognized_time.get(hand_sign_id, 0)
#                 if current_time - last_time >= 2:   #timer to delay gesture sampling, adjust/experiment
#                     last_recognized_time[hand_sign_id] = current_time
#                     if hand_sign_id == 2:  # Point gesture
#                         if (handedness.classification[0].label[0] == "L"):
#                             print("GoBackwards")
#                             # go_bk(allcfs)
#                             go_bk()
#                         else:
#                             print("GoForwards")
#                             # go_fw(allcfs)
#                             go_fw()
#                     elif hand_sign_id == 0:
#                         if(handedness.classification[0].label[0] == "L"):
#                             print("Land")
#                             # land(allcfs)
#                             land()
#                             get_landing_positions()
#                             print("Operations completed. Exiting now.")
#                             sys.exit()
#                             # reset_to_default()
#                         else:
                    
#                             print("TakeOff")
#                             takeoff()
#                     elif hand_sign_id == 1:
#                         if(handedness.classification[0].label[0] == "L"):
#                             print("GoLeft")
#                             # go_left(allcfs)
#                             go_left()
#                         else:
#                             print("GoRight")
#                             # go_right(allcfs)
#                             go_right()
#                     elif hand_sign_id == 3:
#                         if(handedness.classification[0].label[0] == "L"):
#                             print("MoveDown")
#                             # go_down(allcfs)
#                             go_down()
#                         else:
#                             print("MoveUp")
#                             # go_up(allcfs)
#                             go_up()
                        

                            
                            # run_nice_hover()
                        
                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                # debug_image = draw_info_text(
                #     debug_image,
                #     brect,
                #     handedness,
                #     keypoint_classifier_labels[hand_sign_id],
                #     point_history_classifier_labels[most_common_fg_id[0][0]],
                # )
        # else:
        #     point_history.append([0, 0])
        
        debug_image = draw_info(debug_image, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)
        # time.sleep(3)\
        sleep_ms(100) #lowering the camera framerate, adjust/experiment

    cap.release()
    cv.destroyAllWindows()

def sleep_ms(milliseconds):
    start_time = time.monotonic()
    end_time = start_time + milliseconds / 1000.0
    while time.monotonic() < end_time:
        pass


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image


# def draw_info_text(image, brect, handedness, hand_sign_text,
#                    finger_gesture_text):
#     cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
#                  (0, 0, 0), -1)

#     info_text = handedness.classification[0].label[0:]
#     if hand_sign_text != "":
#         info_text = info_text + ':' + hand_sign_text
#     cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
#                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

#     return image


# def draw_point_history(image, point_history):
#     for index, point in enumerate(point_history):
#         if point[0] != 0 and point[1] != 0:
#             cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
#                       (152, 251, 152), 2)

#     return image


def draw_info(image, mode, number):
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    # swarm = Crazyswarm()
    # timeHelper = swarm.timeHelper
    # allcfs = swarm.allcfs
    main()
