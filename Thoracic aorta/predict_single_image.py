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


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default='./Data_folder/images/val/image2.nii')
parser.add_argument("--label", type=str, default='./Data_folder/labels/val/label2.nii')
parser.add_argument("--result", type=str, default='./Data_folder/results_2/val/2.nii', help='path to the .nii result to save')
parser.add_argument("--weights", type=str, default='./best_metric_model.pth', help='network weights to load')
parser.add_argument("--resolution", default=(1.3671875 * 1.2, 1.3671875 * 1.2, 3.0 * 1.2), help='New resolution')
parser.add_argument("--patch_size", type=int, nargs=3, default=(128, 128, 128), help="Input dimension for the generator")
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


def resize(img, new_size, interpolator):
    # img = sitk.ReadImage(img)
    dimension = img.GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = new_size
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
    # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
    # no new labels are introduced.

    return sitk.Resample(img, reference_image, centered_transform, interpolator, 0.0)


def segment(image, label, result, weights, resolution, patch_size):

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if label is not None:
        files = [{"image": image, "label": label}]
    else:
        files = [{"image": image}]

    if label is not None:
        if resolution is not None:
            val_transforms = Compose([
                LoadNiftid(keys=['image', 'label']),
                AddChanneld(keys=['image', 'label']),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-120, a_max=170, b_min=0.0, b_max=1.0, clip=True,
                ),
                NormalizeIntensityd(keys=['image']),
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image', 'label'], pixdim=resolution, mode=('bilinear', 'nearest')),
                ToTensord(keys=['image', 'label'])
            ])
        else:
            val_transforms = Compose([
                LoadNiftid(keys=['image', 'label']),
                AddChanneld(keys=['image', 'label']),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-120, a_max=170, b_min=0.0, b_max=1.0, clip=True,
                ),
                NormalizeIntensityd(keys=['image']),
                ScaleIntensityd(keys=['image']),
                ToTensord(keys=['image', 'label'])
            ])

    else:
        if resolution is not None:
            val_transforms = Compose([
                LoadNiftid(keys=['image']),
                AddChanneld(keys=['image']),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-120, a_max=170, b_min=0.0, b_max=1.0, clip=True,
                ),
                NormalizeIntensityd(keys=['image']),
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image'], pixdim=resolution, mode='bilinear'),
                ToTensord(keys=['image'])
            ])
        else:
            val_transforms = Compose([
                LoadNiftid(keys=['image']),
                AddChanneld(keys=['image']),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-120, a_max=170, b_min=0.0, b_max=1.0, clip=True,
                ),
                NormalizeIntensityd(keys=['image']),
                ScaleIntensityd(keys=['image']),
                ToTensord(keys=['image'])
            ])

    val_ds = monai.data.Dataset(data=files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    dice_metric = DiceMetric(include_background=True, to_onehot_y=False, sigmoid=True, reduction="mean")

    # try to use all the available GPUs
    devices = get_devices_spec(None)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids  # Multi-gpu selector for training

    if len(devices) > 1:

        # build the network
        net = build_net()
        net = net.to(devices[0])

        net = torch.nn.DataParallel(net, device_ids=devices)
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
                # define sliding window size and batch size for windows inference
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                val_outputs = (val_outputs.sigmoid() >= 0.5).float()

        else:
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].cuda(), val_data["label"].cuda()
                # define sliding window size and batch size for windows inference
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                value = dice_metric(y_pred=val_outputs, y=val_labels)
                metric_count += len(value)
                metric_sum += value.item() * len(value)
                val_outputs = (val_outputs.sigmoid() >= 0.5).float()

            metric = metric_sum / metric_count
            print("Evaluation Metric (Dice):", metric)

        result_array = val_outputs.squeeze().data.cpu().numpy()

        if resolution is not None:

            reader = sitk.ImageFileReader()
            reader.SetFileName(image)
            image_itk = reader.Execute()

            res = sitk.GetImageFromArray(np.transpose(result_array, (2, 1, 0)))
            # res.SetDirection(image_itk.GetDirection())
            res.SetSpacing(resolution)
            # image.SetOrigin(image_itk.GetOrigin())
            res = resize(res, (sitk.GetArrayFromImage(image_itk)).shape[::-1], sitk.sitkNearestNeighbor)
            res = (np.rint(sitk.GetArrayFromImage(res)))
            res = sitk.GetImageFromArray(res)
            res.SetDirection(image_itk.GetDirection())
            res.SetOrigin(image_itk.GetOrigin())
            res.SetSpacing(image_itk.GetSpacing())

        else:
            res = from_numpy_to_itk(result_array, image)

        # save label
        writer = sitk.ImageFileWriter()
        writer.SetFileName(result)
        writer.Execute(res)
        print("Saved Result at:", str(result))

        if resolution is not None:

            def dice_coeff(seg, gt):
                """
                function to calculate the dice score
                """
                seg = seg.flatten()
                gt = gt.flatten()
                dice = float(2 * (gt * seg).sum()) / float(gt.sum() + seg.sum())
                return dice

            y_pred = sitk.GetArrayFromImage(sitk.ReadImage(result))
            y_true = sitk.GetArrayFromImage(sitk.ReadImage(label))

            print("Dice after resampling to original resolution:", dice_coeff(y_pred, y_true))


if __name__ == "__main__":

    segment(args.image, args.label, args.result, args.weights, args.resolution, args.patch_size)











