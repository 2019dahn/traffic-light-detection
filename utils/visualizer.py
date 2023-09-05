"""
    Visualizer class for showing images, bounding boxes, and labels_json
"""

import cv2
import json

DATA_PATH = 'TrafficLightData/data/'


# Main function to visualize the image
def visualize_image(folder_number: int, image_number: int, data: str = 'Training'):
    """
        folder_number -> 0 ~ 999
        image_number -> 1 ~ 49
        data -> 'Training' or 'Validation'
    """

    # Check if the folder_number and image_number are valid
    if not (0 <= folder_number <= 999 and 1 <= image_number <= 49) or not (data == 'Training' or data == 'Validation'):
        print("Invalid input")
        return

    image_path = DATA_PATH + f'{data}/images/TS_Clip_{str(folder_number).zfill(4)}_Camera_Camera_Front/' + \
                 f'512_ND_{str(folder_number).zfill(4)}_CF_{str(image_number).zfill(3)}.jpg'

    json_path = DATA_PATH + f'{data}/labels_json/TL_Clip_{str(folder_number).zfill(4)}_Camera_Camera_Front/' + \
                f'512_ND_{str(folder_number).zfill(4)}_CF_{str(image_number).zfill(3)}.json'

    # Read the data
    image_file = cv2.imread(image_path)
    json_file = json.load(open(json_path, 'r'))

    # Draw the bounding boxes
    for box in json_file['objects']:
        x1, y1 = map(int, box['data'][0])
        x2, y2 = map(int, box['data'][1])
        class_name = box['class_name']
        signal = box['attribute']['signal']
        color = (0, 255, 0) if signal == 'green' \
            else (0, 0, 255) if signal == 'red' \
            else (0, 0, 0)

        # Draw the bounding box and the label
        cv2.putText(image_file, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.rectangle(image_file, (x1, y1), (x2, y2), color, 2)

    # Show the image
    cv2.imshow('image', image_file)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    FOLDER_NUMBER = 0
    IMAGE_NUMBER = 4

    try:
        visualize_image(FOLDER_NUMBER, IMAGE_NUMBER)
    except FileNotFoundError:
        print("File not found")
