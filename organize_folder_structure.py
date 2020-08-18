import os
import re
import argparse
import SimpleITK as sitk


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
parser.add_argument('--images', default='./Data_folder/imagesTr', help='path to the images a (early frames)')
parser.add_argument('--labels', default='./Data_folder/labelsTr', help='path to the images b (late frames)')
parser.add_argument('--split', default=7, help='number of images for testing')
args = parser.parse_args()

if __name__ == "__main__":

    list_images = lstFiles(args.images)
    list_labels = lstFiles(args.labels)


    os.mkdir('./Data_folder/images')
    os.mkdir('./Data_folder/labels')

    if not os.path.isdir('./Data_folder/images/train'):
        os.mkdir('./Data_folder/images/train/')

    if not os.path.isdir('./Data_folder/images/val'):
        os.mkdir('./Data_folder/images/val')

    if not os.path.isdir('./Data_folder/labels/train'):
        os.mkdir('./Data_folder/labels/train')

    if not os.path.isdir('./Data_folder/labels/val'):
        os.mkdir('./Data_folder/labels/val')

    for i in range(len(list_images)-int(args.split)):

        a = list_images[int(args.split)+i]
        b = list_labels[int(args.split)+i]

        print(a)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image_directory = os.path.join('./Data_folder/images/train', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/labels/train', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)


    for i in range(int(args.split)):

        a = list_images[i]
        b = list_labels[i]

        print(a)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image_directory = os.path.join('./Data_folder/images/val', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/labels/val', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

