#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from utils import *
import argparse
from networks import build_net, build_UNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import NiftiSaver, create_test_image_3d, list_data_collate, decollate_batch
from monai.transforms import (EnsureType, Compose, LoadImaged, AddChanneld, Transpose,Activations,AsDiscrete, RandGaussianSmoothd, CropForegroundd, SpatialPadd,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, RandZoomd,
                              Spacingd, Orientationd, Resized, ThresholdIntensityd, RandShiftIntensityd, BorderPadd, RandGaussianNoised, RandAdjustContrastd,NormalizeIntensityd,RandFlipd)


def segment(image, label, result, weights, resolution, patch_size, network, gpu_ids):

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if label is not None:
        uniform_img_dimensions_internal(image, label, True)
        files = [{"image": image, "label": label}]
    else:
        files = [{"image": image}]

    # original size, size after crop_background, cropped roi coordinates, cropped resampled roi size
    original_shape, crop_shape, coord1, coord2, resampled_size, original_resolution = statistics_crop(image, resolution)

    # -------------------------------

    if label is not None:
        if resolution is not None:

            val_transforms = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=['image', 'label']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # Threshold CT
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                CropForegroundd(keys=['image', 'label'], source_key='image'),  # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image', 'label'], pixdim=resolution, mode=('bilinear', 'nearest')),  # resolution

                SpatialPadd(keys=['image', 'label'], spatial_size=patch_size, method= 'end'),
                ToTensord(keys=['image', 'label'])])
        else:

            val_transforms = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=['image', 'label']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # Threshold CT
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                CropForegroundd(keys=['image', 'label'], source_key='image'),  # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),

                SpatialPadd(keys=['image', 'label'], spatial_size=patch_size, method='end'),  # pad if the image is smaller than patch
                ToTensord(keys=['image', 'label'])])

    else:
        if resolution is not None:

            val_transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # Threshold CT
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                CropForegroundd(keys=['image'], source_key='image'),  # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image'], pixdim=resolution, mode=('bilinear')),  # resolution

                SpatialPadd(keys=['image'], spatial_size=patch_size, method= 'end'),  # pad if the image is smaller than patch
                ToTensord(keys=['image'])])
        else:

            val_transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # Threshold CT
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                CropForegroundd(keys=['image'], source_key='image'),  # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),

                SpatialPadd(keys=['image'], spatial_size=patch_size, method='end'), # pad if the image is smaller than patch
                ToTensord(keys=['image'])])

    val_ds = monai.data.Dataset(data=files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, collate_fn=list_data_collate, pin_memory=False)

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

    if gpu_ids != '-1':

        # try to use all the available GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        device = torch.device("cpu")

    # build the network
    if network == 'nnunet':
        net = build_net()  # nn build_net
    elif network == 'unetr':
        net = build_UNETR() # UneTR

    net = net.to(device)

    if gpu_ids == '-1':

        net.load_state_dict(new_state_dict_cpu(weights))

    else:

        net.load_state_dict(new_state_dict(weights))

    # define sliding window size and batch size for windows inference
    roi_size = patch_size
    sw_batch_size = 4

    net.eval()
    with torch.no_grad():

        if label is None:
            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

        else:
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            print("Evaluation Metric (Dice):", metric)

        result_array = val_outputs[0].squeeze().data.cpu().numpy()
        # Remove the pad if the image was smaller than the patch in some directions
        result_array = result_array[0:resampled_size[0],0:resampled_size[1],0:resampled_size[2]]

        # resample back to the original resolution
        if resolution is not None:

            result_array_np = np.transpose(result_array, (2, 1, 0))
            result_array_temp = sitk.GetImageFromArray(result_array_np)
            result_array_temp.SetSpacing(resolution)

            # save temporary label
            writer = sitk.ImageFileWriter()
            writer.SetFileName('temp_seg.nii')
            writer.Execute(result_array_temp)

            files = [{"image": 'temp_seg.nii'}]

            files_transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                Spacingd(keys=['image'], pixdim=original_resolution, mode=('nearest')),
                Resized(keys=['image'], spatial_size=crop_shape, mode=('nearest')),
            ])

            files_ds = Dataset(data=files, transform=files_transforms)
            files_loader = DataLoader(files_ds, batch_size=1, num_workers=0)

            for files_data in files_loader:
                files_images = files_data["image"]

                res = files_images.squeeze().data.numpy()

            result_array = np.rint(res)

            os.remove('./temp_seg.nii')

        # recover the cropped background before saving the image
        empty_array = np.zeros(original_shape)
        empty_array[coord1[0]:coord2[0],coord1[1]:coord2[1],coord1[2]:coord2[2]] = result_array

        result_seg = from_numpy_to_itk(empty_array, image)

        # save label
        writer = sitk.ImageFileWriter()
        writer.SetFileName(result)
        writer.Execute(result_seg)
        print("Saved Result at:", str(result))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default='./Data_folder/T2/3.nii', help='source image' )
    parser.add_argument("--label", type=str, default='./Data_folder/T2_labels/3.nii', help='source label, if you want to compute dice. None for new case')
    parser.add_argument("--result", type=str, default='./Data_folder/test_0.nii', help='path to the .nii result to save')
    parser.add_argument("--weights", type=str, default='./best_metric_model.pth', help='network weights to load')
    parser.add_argument("--resolution", default=[0.7, 0.7, 3], help='Resolution used in training phase')
    parser.add_argument("--patch_size", type=int, nargs=3, default=(256, 256, 16), help="Input dimension for the generator, same of training")
    parser.add_argument('--network', default='unetr', help='nnunet, unetr')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    args = parser.parse_args()

    segment(args.image, args.label, args.result, args.weights, args.resolution, args.patch_size, args.network, args.gpu_ids)













