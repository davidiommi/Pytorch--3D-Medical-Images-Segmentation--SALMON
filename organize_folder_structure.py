import os
import re
import argparse
import SimpleITK as sitk
import numpy as np
import random
from utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='./Data_folder/CT', help='path to the images')
    parser.add_argument('--labels', default='./Data_folder/CT_label', help='path to the labels')
    parser.add_argument('--split_val', default=30, help='number of images for validation')
    parser.add_argument('--split_test', default=30, help='number of images for testing')
    parser.add_argument('--resolution', default=[2.25, 2.25, 3], help='New Resolution, if you want to resample the data')
    args = parser.parse_args()

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

        print('train',i, a,b)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear', fill_value=0)
        image, label = uniform_img_dimensions(image, label, nearest=True)
        label = gaussian2(label)

        image_directory = os.path.join('./Data_folder/images/train', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/labels/train', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

    for i in range(int(args.split_val)):

        a = list_images[int(args.split_test)+i]
        b = list_labels[int(args.split_test)+i]

        print('val',i, a,b)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear', fill_value=0)
        image, label = uniform_img_dimensions(image, label, nearest=True)
        label = gaussian2(label)

        image_directory = os.path.join('./Data_folder/images/val', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/labels/val', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

    for i in range(int(args.split_test)):

        a = list_images[i]
        b = list_labels[i]

        print('test',i,a,b)

        label = sitk.ReadImage(b)
        image = sitk.ReadImage(a)

        image = resample_sitk_image(image, spacing=args.resolution, interpolator='linear', fill_value=0)
        image, label = uniform_img_dimensions(image, label, nearest=True)
        label = gaussian2(label)

        image_directory = os.path.join('./Data_folder/images/test', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/labels/test', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

