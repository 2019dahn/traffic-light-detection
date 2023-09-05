# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.common import output_tensor
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from numpy import copy
import time

def detect():
    default_model_dir = '../all_models'
    default_model = 'traffic-yolov5n-int8_edgetpu.tflite'
    default_labels = 'traffic_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='classifier score threshold')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default='./out.mp4')
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--fps', type=int, default=10)

    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    print("Labels:\t\t\t", labels)
    inference_size = input_size(interpreter)
    print("Inference size:\t\t", inference_size)
    
    if args.input:
        cap = cv2.VideoCapture(args.input)
        print("Video read from:\t", args.input)
    else:
        cap = cv2.VideoCapture(args.camera_idx)
        print("Video read from camera:\t", args.camera_idx)

    if cap.isOpened():
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        internal_fps = cap.get(cv2.CAP_PROP_FPS)
        print("Video FPS:\t\t", args.fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, internal_fps, (frame_width, frame_height))
    frames = args.fps * args.length
    i, j = 0, 0

    while cap.isOpened() and frames > 0:
        ret, frame = cap.read()
        
        # Only process every nth frame
        if i * (internal_fps / args.fps) > j:
          j += 1
          # Writes frame without bounding box
          out.write(frame)
          continue
        
        if not ret:
            break
        cv2_im = frame
        
        print(f"Frame {i}... ", end="")
        start = time.time()

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        prediction = output_tensor(interpreter, 0)[0]
        
        max_confidence_box = max(prediction, key=lambda x: x[4])
        
        max_confidence_box = xywh2xyxy(max_confidence_box)
        
        obj = TrafficLightObject(max_confidence_box[:4], "traffic light", max_confidence_box[4] / 640)
        if obj.score > args.threshold:
            frame = append_objs_to_img(cv2_im, inference_size, obj, labels)

        out.write(frame)
        frames -= 1
        
        print("done in", time.time() - start, "seconds.")
        i += 1
        j += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, obj, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]

    bbox = rescale(obj.bbox, scale_x, scale_y)
    xmin, ymin, xmax, ymax = bbox
    x0, y0 = int(xmin), int(ymin)
    x1, y1 = int(xmax), int(ymax)

    label = '{} {}%'.format(labels.get(obj.id, obj.id), int(obj.score * 100))

    cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    return cv2_im
  
def xywh2xyxy(x):
  # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2 # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2# top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2# bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2 # bottom right y
    return y / 255 * 640
  
def rescale(bbox, sx, sy):
  xmin, ymin, xmax, ymax = bbox
  return [
    sx * xmin,
    sy * ymin,
    sx * xmax,
    sy * ymax
  ]
  
class TrafficLightObject():
  
  def __init__(self, bbox, label, score):
    self.id = 0
    self.bbox = bbox
    self.label = label
    self.score = score

if __name__ == '__main__':
    detect()
