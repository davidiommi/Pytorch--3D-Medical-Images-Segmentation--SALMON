import os
import re
import argparse
import SimpleITK as sitk
import random


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def lstFiles(Path):

    images_list = []  # create an empty list, the raw image data files is stored here
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".nii.gz" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".nii" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".mhd" in filename.lower():
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)

    return images_list


parser = argparse.ArgumentParser()
parser.add_argument('--images', default='./Data_folder/imagesTr', help='path to the images')
parser.add_argument('--labels', default='./Data_folder/labelsTr', help='path to the labels')
parser.add_argument('--split_val', default=3, help='number of images for validation')
parser.add_argument('--split_test', default=2, help='number of images for testing')
args = parser.parse_args()

if __name__ == "__main__":

    list_images = lstFiles(args.images)
    list_labels = lstFiles(args.labels)

    mapIndexPosition = list(zip(list_images, list_labels))  # shuffle order list
    random.shuffle(mapIndexPosition)
    list_images, list_labels = zip(*mapIndexPosition)

    os.mkdir('./Data_folder/images')
    os.mkdir('./Data_folder/labels')

    # 1
    if not os.path.isdir('./Data_folder/images/train'):
        os.mkdir('./Data_folder/images/train/')
    # 2
    if not os.path.isdir('./Data_folder/images/val'):
        os.mkdir('./Data_folder/images/val')

    # 3
    if not os.path.isdir('./Data_folder/images/test'):
        os.mkdir('./Data_folder/images/test')

    # 4
    if not os.path.isdir('./Data_folder/labels/train'):
        os.mkdir('./Data_folder/labels/train')

    # 5
    if not os.path.isdir('./Data_folder/labels/val'):
        os.mkdir('./Data_folder/labels/val')

    # 6
    if not os.path.isdir('./Data_folder/labels/test'):
        os.mkdir('./Data_folder/labels/test')

    for i in range(len(list_images)-int(args.split_test + args.split_val)):

        a = list_images[int(args.split_test + args.split_val)+i]
        b = list_labels[int(args.split_test + args.split_val)+i]

        print(a)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image_directory = os.path.join('./Data_folder/images/train', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/labels/train', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

    for i in range(int(args.split_val)):

        a = list_images[int(args.split_test)+i]
        b = list_labels[int(args.split_test)+i]

        print(a)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image_directory = os.path.join('./Data_folder/images/val', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/labels/val', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

    for i in range(int(args.split_test)):

        a = list_images[i]
        b = list_labels[i]

        print(a)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image_directory = os.path.join('./Data_folder/images/test', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/labels/test', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

