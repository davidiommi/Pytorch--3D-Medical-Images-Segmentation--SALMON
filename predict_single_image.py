#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from train import *
import argparse
from networks import *
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import NiftiSaver, create_test_image_3d, list_data_collate
from collections import OrderedDict
from organize_folder_structure import resize, resample_sitk_image, uniform_img_dimensions


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default='./Data_folder/images/test/image0.nii')
parser.add_argument("--label", type=str, default='./Data_folder/labels/test/label0.nii')
parser.add_argument("--result", type=str, default='./Data_folder/test.nii', help='path to the .nii result to save')
parser.add_argument("--weights", type=str, default='./best_metric_model.pth', help='network weights to load')
parser.add_argument("--resolution", default=[3,3,3], help='New resolution')
parser.add_argument("--patch_size", type=int, nargs=3, default=(128, 128, 64), help="Input dimension for the generator")
parser.add_argument('--gpu_ids', type=str, default='2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
args = parser.parse_args()


def new_state_dict(file_name):
    state_dict = torch.load(file_name)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == 'module':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def from_numpy_to_itk(image_np, image_itk):

    # read image file
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_itk)
    image_itk = reader.Execute()

    image_np = np.transpose(image_np, (2, 1, 0))
    image = sitk.GetImageFromArray(image_np)
    image.SetDirection(image_itk.GetDirection())
    image.SetSpacing(image_itk.GetSpacing())
    image.SetOrigin(image_itk.GetOrigin())
    return image


# function to keep track of the cropped area and coordinates
def statistics_crop(image, resolution):

    files = [{"image": image}]

    reader = sitk.ImageFileReader()
    reader.SetFileName(image)
    image_itk = reader.Execute()
    original_resolution = image_itk.GetSpacing()


    # original size
    transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        ToTensord(keys=['image'])])
    data = monai.data.Dataset(data=files, transform=transforms)
    loader = DataLoader(data, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
    loader = monai.utils.misc.first(loader)
    im, = (loader['image'][0])
    vol = im.numpy()
    original_shape = vol.shape

    # cropped foreground size
    transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        CropForegroundd(keys=['image'], source_key='image', start_coord_key='foreground_start_coord',
                        end_coord_key='foreground_end_coord', ),  # crop CropForeground
        ToTensord(keys=['image', 'foreground_start_coord', 'foreground_end_coord'])])
    data = monai.data.Dataset(data=files, transform=transforms)
    loader = DataLoader(data, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
    loader = monai.utils.misc.first(loader)
    im, coord1, coord2 = (loader['image'][0], loader['foreground_start_coord'][0], loader['foreground_end_coord'][0])
    vol = im[0].numpy()
    coord1 = coord1.numpy()
    coord2 = coord2.numpy()
    crop_shape = vol.shape

    if resolution is not None:

        transforms = Compose([
            LoadImaged(keys=['image']),
            AddChanneld(keys=['image']),
            CropForegroundd(keys=['image'], source_key='image'),  # crop CropForeground
            Spacingd(keys=['image'], pixdim=resolution, mode=('bilinear')),  # resolution
            ToTensord(keys=['image'])])

        data = monai.data.Dataset(data=files, transform=transforms)
        loader = DataLoader(data, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
        loader = monai.utils.misc.first(loader)
        im, = (loader['image'][0])
        vol = im.numpy()
        resampled_size = vol.shape

    else:

        resampled_size = original_shape

    return original_shape, crop_shape, coord1, coord2, resampled_size, original_resolution


def segment(image, label, result, weights, resolution, patch_size):

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if label is not None:
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
                CropForegroundd(keys=['image', 'label'], source_key='image'),  # crop CropForeground
                ThresholdIntensityd(keys=['image'], threshold=-350, above=True, cval=-350),  # Threshold CT
                ThresholdIntensityd(keys=['image'], threshold=350, above=False, cval=350),

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image', 'label'], pixdim=resolution, mode=('bilinear', 'nearest')),  # resolution

                SpatialPadd(keys=['image', 'label'], spatial_size=patch_size, method= 'end'),
                ToTensord(keys=['image', 'label'])])
        else:

            val_transforms = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=['image', 'label']),
                CropForegroundd(keys=['image', 'label'], source_key='image'),  # crop CropForeground
                ThresholdIntensityd(keys=['image'], threshold=-350, above=True, cval=-350),  # Threshold CT
                ThresholdIntensityd(keys=['image'], threshold=350, above=False, cval=350),

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),

                SpatialPadd(keys=['image', 'label'], spatial_size=patch_size, method='end'),  # pad if the image is smaller than patch
                ToTensord(keys=['image', 'label'])])

    else:
        if resolution is not None:

            val_transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),

                CropForegroundd(keys=['image'], source_key='image'),  # crop CropForeground
                ThresholdIntensityd(keys=['image'], threshold=-350, above=True, cval=-350),  # Threshold CT
                ThresholdIntensityd(keys=['image'], threshold=350, above=False, cval=350),

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image'], pixdim=resolution, mode=('bilinear')),  # resolution

                SpatialPadd(keys=['image'], spatial_size=patch_size, method= 'end'),  # pad if the image is smaller than patch
                ToTensord(keys=['image'])])
        else:

            val_transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                CropForegroundd(keys=['image'], source_key='image'),  # crop CropForeground
                ThresholdIntensityd(keys=['image'], threshold=-350, above=True, cval=-350),  # Threshold CT
                ThresholdIntensityd(keys=['image'], threshold=350, above=False, cval=350),

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),

                SpatialPadd(keys=['image'], spatial_size=patch_size, method='end'), # pad if the image is smaller than patch
                ToTensord(keys=['image'])])

    val_ds = monai.data.Dataset(data=files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

    # try to use all the available GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids  # Multi-gpu selector for training

    if args.gpu_ids != '-1':
        num_gpus = len(args.gpu_ids.split(','))
    else:
        num_gpus = 0
    print('number of GPU:', num_gpus)

    if num_gpus > 1:

        # build the network
        net = build_net().cuda()

        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(weights))

    else:

        net = build_net().cuda()
        net.load_state_dict(new_state_dict(weights))

    # define sliding window size and batch size for windows inference
    roi_size = patch_size
    sw_batch_size = 4

    net.eval()
    with torch.no_grad():

        if label is None:
            for val_data in val_loader:
                val_images = val_data["image"].cuda()
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                val_outputs = post_trans(val_outputs)
                # val_outputs = (val_outputs.sigmoid() >= 0.5).float()

        else:
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].cuda(), val_data["label"].cuda()
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                val_outputs = post_trans(val_outputs)
                value, _ = dice_metric(y_pred=val_outputs, y=val_labels)
                metric_count += len(value)
                metric_sum += value.item() * len(value)
                # val_outputs = (val_outputs.sigmoid() >= 0.5).float()

            metric = metric_sum / metric_count
            print("Evaluation Metric (Dice):", metric)

        result_array = val_outputs.squeeze().data.cpu().numpy()

        result_array = result_array[0:resampled_size[0],0:resampled_size[1],0:resampled_size[2]]

        if resolution is not None:

            result_array_np = np.transpose(result_array, (2, 1, 0))
            result_array_temp = sitk.GetImageFromArray(result_array_np)
            result_array_temp.SetSpacing(resolution)
            label = resample_sitk_image(result_array_temp, spacing=original_resolution, interpolator='nearest')
            res = resize(label, crop_shape, sitk.sitkNearestNeighbor)

            result_array = np.transpose(np.rint(sitk.GetArrayFromImage(res)), axes=(2, 1, 0))

        empty_array = np.zeros(original_shape)

        empty_array[coord1[0]:coord2[0],coord1[1]:coord2[1],coord1[2]:coord2[2]] = result_array

        result_seg = from_numpy_to_itk(empty_array, image)

        # save label
        writer = sitk.ImageFileWriter()
        writer.SetFileName(result)
        writer.Execute(result_seg)
        print("Saved Result at:", str(result))


if __name__ == "__main__":

    segment(args.image, args.label, args.result, args.weights, args.resolution, args.patch_size)













