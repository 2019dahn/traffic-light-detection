# Traffic Light Detector

Detects traffic light and determines which color it is.

Only available in Edge TPU.

<img width="768" alt="image" src="https://github.com/2019dahn/traffic-light-detection/assets/105447127/37915c34-4be0-4921-b38d-3d264ba2d64d">

### 1. Detection 🧐

Model: YOLOv5

Dataset: 2400 for training, 600 for validation ([Link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71579))

Label: Only 1 (visible vehicular light, regardless of the color)

### 2. Tracking ↯

Algorithm: OpenCV CSRT Tracker

Modified the algorithm to keep on tracking the object even when the light color changes.

### 3. Color Classification 🌈

Self made algorithm to determine whether the traffic light is green or red.

Used traditional vision techniques for better operation speed.

## Demo

![demo (1)](https://github.com/2019dahn/traffic-light-detection/assets/105447127/24dfe893-4a1b-4560-9614-25bb37ceccf5)


