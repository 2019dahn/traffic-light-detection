"""
    Title: Detect, Track, Classify
    Author: tfrecord3
    This file is a combined code of detect_traffic_light_vid.py and hue_traffic_signal_test.py
"""

import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.common import output_tensor
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from detect_traffic_light_img import xywh2xyxy, TrafficLightObject, append_objs_to_img, rescale

def main():
    default_model_dir = '../all_models'
    default_model = 'traffic-yolov5n-int8_edgetpu.tflite'
    default_labels = 'traffic_labels.txt'

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path', default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path', default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.4, help='classifier score threshold')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default='./out.mp4')
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--fps', type=int, default=10)

    args = parser.parse_args()

    # Load model
    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    print("Labels:\t\t\t", labels)
    inference_size = input_size(interpreter)
    print("Inference size:\t\t", inference_size)

    # Read video (input from video file)
    if args.input:
        cap = cv2.VideoCapture(args.input)
        print("Video read from:\t", args.input)
    # Read video (input from camera)
    else:
        cap = cv2.VideoCapture(args.camera_idx)
        print("Video read from camera:\t", args.camera_idx)

    if cap.isOpened():
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        internal_fps = cap.get(cv2.CAP_PROP_FPS)
        print("Video FPS:\t\t", args.fps)

    # 트랙커 객체 생성자 함수 리스트
    trackers = [cv2.TrackerCSRT_create,
                cv2.TrackerMIL_create,
                cv2.TrackerKCF_create]
    trackerIdx = 0  # 트랙커 생성자 함수 선택 인덱스
    tracker = None
    isFirst = True

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, internal_fps, (frame_width, frame_height))
    frames = args.fps * args.length
    i, j = 0, 0
    bounding_box = None     # (x, y, w, h)

    # Main loop
    while cap.isOpened() and frames > 0:
        ret, frame = cap.read()

        if not ret:
            print('Cannot read video file')
            break

        # No bounding box yet
        if not tracker:
            # Only process every nth frame
            if i * (internal_fps / args.fps) > j:
                j += 1
                # Writes frame without bounding box
                out.write(frame)
                continue
            else:
                print("Detecting traffic light...")
                bounding_box = detect_traffic_light(frame, interpreter, inference_size, labels, out, args.threshold)
                if bounding_box:
                    print("Traffic light found:\t\t", list(map(int, bounding_box)))
                    tracker = trackers[trackerIdx]()  # 트랙커 객체 생성
                    isInit = tracker.init(frame, bounding_box)
                i, j = i + 1, j + 1

        # Bounding box found
        else:
            # Process every frame
            ok, bounding_box = track_traffic_light(frame, tracker)
            if ok:
                color = classify_traffic_light(frame, bounding_box)
                write_frame_to_video(frame, out, color, bounding_box)
                if color == "fail":
                    print("Traffic light lost. Failed to classify color.")
                    tracker = None
                else:
                    print("Traffic light color:\t\t", color)
            # Failed to track
            else:
                print("Traffic light lost. Failed to track.")
                tracker = None

        # Exit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_traffic_light(frame, interpreter, inference_size, labels, out, threshold):
    cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

    run_inference(interpreter, cv2_im_rgb.tobytes())
    prediction = output_tensor(interpreter, 0)[0]

    max_confidence_box = max(prediction, key=lambda x: x[4])
    max_confidence_box = xywh2xyxy(max_confidence_box)
    obj = TrafficLightObject(bbox=max_confidence_box[:4], label="traffic light", score=max_confidence_box[4] / 640)
    frame = append_objs_to_img(frame, inference_size, obj, labels)
    out.write(frame)

    cv2.destroyAllWindows()

    if obj.score > threshold:
        # Returns x0, y0, x1, y1
        sx = frame.shape[1] / inference_size[0]
        sy = frame.shape[0] / inference_size[1]
        x0, y0, x1, y1 = rescale(max_confidence_box[:4], sx, sy)
        return x0, y0, x1 - x0, y1 - y0  # x, y, w, h
    else:
        return None


def track_traffic_light(frame, tracker):
    # Update tracker to get new bounding box in subsequent frames
    return tracker.update(frame)


def classify_traffic_light(frame, bounding_box):
    # crop bbox
    x1, y1, w, h = map(int, bounding_box)
    x2 = x1 + w
    y2 = y1 + h
    cropped_image = frame[int(y1): int(y2), int(x1):int(x2)]

    # classify cropped image
    red_pxl_counter, green_pxl_counter, total_pxls = 0, 0, 0
    bg_counter = 0

    change_threshold = False
    red_threshold = 0.02
    green_threshold = 0.02

    for i, col in enumerate(cropped_image):
        for j, pxl in enumerate(col):

            total_pxls += 1

            blue, green, red = pxl[0], pxl[1], pxl[2]

            # skip background white pxl
            if red > 200 and green > 200 and blue > 200:
                bg_counter += 1
                # if there exists background ~white region
                if bg_counter > 150 and green_pxl_counter > 150 or red_pxl_counter > 150:
                    change_threshold = True
                continue

            if red > 150 and blue < 150:
                red_pxl_counter += 1

            if green > 170 and red < 150:
                green_pxl_counter += 1

    # in case of bbox_not_tight
    if change_threshold == True:
        red_threshold = 0.04
        green_threshold = 0.04

    if red_pxl_counter > red_threshold * total_pxls:
        return "red"
    elif green_pxl_counter > green_threshold * total_pxls:
        return "green"
    else:
        return "fail"


def write_frame_to_video(frame, out, color, bounding_box):
    # Bounding box is x y w h
    x, y, w, h = map(int, bounding_box)
    if color == "red":
        bounding_box_color = (0, 0, 255)
    elif color == "green":
        bounding_box_color = (0, 255, 0)
    else:
        bounding_box_color = (0, 0, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), bounding_box_color, 2)
    out.write(frame)


if __name__ == '__main__':
    main()
