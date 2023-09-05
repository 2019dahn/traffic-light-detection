import os
import shutil

ROOT_PATH = '../datasets/traffic_light_data_3000/training'


# Check if all images have corresponding label files
# Check if all label files have corresponding image files
def check_image_and_labels():
    image_list = os.listdir(ROOT_PATH + '/images')
    label_list = os.listdir(ROOT_PATH + '/labels')
    a, b = 0, 0

    for image in image_list:
        try:
            if image[:-4] + '.txt' not in label_list:
                print("Image file without label file: ", image)
                a += 1
        except ValueError:
            print("Invalid image file: ", image)
            a += 1

    for label in label_list:
        try:
            if label[:-4] + '.jpg' not in image_list:
                FOLDER_NUM = int(label.split('_')[2])
                FILE_NUM = int(label.split('_')[4][:-4])
                print(FOLDER_NUM, FILE_NUM)
                b += 1
        except ValueError:
            print("Invalid label file: ", label)
            b += 1

    print("Total number of image files without label files: ", a)
    print("Total number of label files without image files: ", b)


# Rename all files in {FOLDER_NUM}_{FILE_NUM}.jpg format
def rename_files():
    image_list = os.listdir(ROOT_PATH + '/images')
    label_list = os.listdir(ROOT_PATH + '/labels')

    for image in image_list:
        folder_num = image.split('_')[2]
        file_num = image.split('_')[4][:-4]
        print(folder_num, file_num)
        os.rename(ROOT_PATH + '/images/' + image, ROOT_PATH + '/images/' + folder_num + '_' + file_num + '.jpg')

    for label in label_list:
        folder_num = label.split('_')[2]
        file_num = label.split('_')[4][:-4]
        print(folder_num, file_num)
        os.rename(ROOT_PATH + '/labels/' + label, ROOT_PATH + '/labels/' + folder_num + '_' + file_num + '.txt')


# Move files from raw dataset to the new dataset
def move_files():

    FROM_PATH = '../../traffic_light_data_raw/Training/labels_json'
    TO_PATH = '../datasets/traffic_light_data_3000/training/labels_json'

    list_of_source_folders = os.listdir(FROM_PATH)

    for folder in list_of_source_folders:

        if folder == '.DS_Store':
            continue

        folder_num = int(folder.split('_')[2])
        print("Copying files from folder", folder_num, end="...\n")

        # Pick the first, intermediate, and the last image from each folder
        file_list = os.listdir(FROM_PATH + '/' + folder)
        file_list.sort()
        file_first, file_last = file_list[0], file_list[-1]
        file_mid = file_list[len(file_list) // 2]

        # Move the first and the last image
        shutil.copyfile(FROM_PATH + '/' + folder + '/' + file_first, TO_PATH + '/' + file_first)
        shutil.copyfile(FROM_PATH + '/' + folder + '/' + file_mid, TO_PATH + '/' + file_mid)
        shutil.copyfile(FROM_PATH + '/' + folder + '/' + file_last, TO_PATH + '/' + file_last)

    print("Done")


# Remove DS_Store files
def remove_ds_store():
    image_list = os.listdir(ROOT_PATH + '/images')
    label_list = os.listdir(ROOT_PATH + '/labels')

    for image in image_list:
        if image == '.DS_Store':
            os.remove(ROOT_PATH + '/images/' + image)
            print("Removed", image)

    for label in label_list:
        if label == '.DS_Store':
            os.remove(ROOT_PATH + '/labels/' + label)
            print("Removed", label)


if __name__ == '__main__':
    # check_image_and_labels()
    # rename_files()
    # move_files()
    # remove_ds_store()
    pass
