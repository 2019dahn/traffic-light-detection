"""
    This file is used to convert the json label file to txt file.
"""

import json
import os

# Class to store the label information
class ImageLabel:

    def __init__(self, class_id: int, x: float, y: float, w: float, h: float):
        self.class_id = class_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# Main function to convert json to txt
def json_to_txt(input_path, output_path):
    """
        input_path: The path to the json file
        output_path: The path to the output txt file
    """

    json_file = open(input_path, 'r')
    json_data = json.load(json_file)

    width, height = map(int, json_data['image_size'])
    labels = []

    for box in json_data['objects']:
        x1, y1 = map(int, box['data'][0])
        x2, y2 = map(int, box['data'][1])
        class_name = box['class_name']
        signal = box['attribute']['signal']

        if class_name == 'vehicular_signal':
            # Only when distingushing between red and green

            # if signal == 'red':
            #     class_id = 0
            # elif signal == 'green':
            #     class_id = 1
            # else:
            #     continue

            class_id = 0
        else:
            continue

        # Normalize the coordinates
        center_x, center_y = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        box_width, box_height = (x2 - x1) / width, (y2 - y1) / height

        labels.append(ImageLabel(class_id, center_x, center_y, box_width, box_height))

    # Write the data to the txt file
    with open(output_path, 'w') as f:
        for label in labels:
            f.write(f'{label.class_id} {label.x: .6f} {label.y: .6f} {label.w: .6f} {label.h: .6f}\n')


# For json files with deprecated format
def json_to_txt2(input_path, output_path):

    json_file = open(input_path, 'r')
    json_data = json.load(json_file)

    width, height = int(json_data['imageWidth']), int(json_data['imageHeight'])
    labels = []

    for box in json_data['shapes']:
        x1, y1 = map(int, box['points'][0])
        x2, y2 = map(int, box['points'][1])
        class_name = box['label']
        signal = box['flags']['signal']

        if class_name == 'vehicular_signal':
            # Only when distingushing between red and green

            # if signal == 'red':
            #     class_id = 0
            # elif signal == 'green':
            #     class_id = 1
            # else:
            #     continue

            class_id = 0
        else:
            continue

        # Normalize the coordinates
        center_x, center_y = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        box_width, box_height = (x2 - x1) / width, (y2 - y1) / height

        labels.append(ImageLabel(class_id, center_x, center_y, box_width, box_height))

    # Write the data to the txt file
    with open(output_path, 'w') as f:
        for label in labels:
            f.write(f'{label.class_id} {label.x: .6f} {label.y: .6f} {label.w: .6f} {label.h: .6f}\n')


# Main function
if __name__ == '__main__':

    ROOT_PATH = '../datasets/traffic_light_data_3000/training/'

    for json_file in os.listdir(ROOT_PATH + 'labels_json'):

            input_path = ROOT_PATH + f'labels_json/{json_file}'

            output_path = ROOT_PATH + f'labels/{json_file[:-5]}.txt'

            # Convert and save the txt file
            try:
                json_to_txt(input_path, output_path)
            except FileNotFoundError:
                print(f"File not found for input path: {input_path}")
                continue
            except KeyError:
                try:
                    json_to_txt2(input_path, output_path)
                except KeyError:
                    print(f"KeyError for input path: {input_path}")
                    continue
