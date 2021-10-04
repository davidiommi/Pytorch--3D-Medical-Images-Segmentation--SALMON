import os
import re
import argparse
import SimpleITK as sitk
import numpy as np
import random


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


def resample_sitk_image(sitk_image, spacing=None, interpolator=None, fill_value=0):
    # https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }

    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1:  # 8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), \
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()

    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   sitk.Transform(),
                                                   sitk_interpolator,
                                                   orig_origin,
                                                   new_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)

    return resampled_sitk_image


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
            elif ".nrrd" in filename.lower():
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)

    return images_list


def uniform_img_dimensions(image, label):

    image_array = sitk.GetArrayFromImage(image)
    image_array = np.transpose(image_array, axes=(2, 1, 0))  # reshape array from itk z,y,x  to  x,y,z
    image_shape = image_array.shape

    label = resample_sitk_image(label, spacing=image.GetSpacing(), interpolator='nearest')
    res = resize(label,image_shape,sitk.sitkNearestNeighbor)
    res = (np.rint(sitk.GetArrayFromImage(res)))
    res = sitk.GetImageFromArray(res.astype('uint8'))
    res.SetDirection(image.GetDirection())
    res.SetOrigin(image.GetOrigin())
    res.SetSpacing(image.GetSpacing())
    print(res.GetSize())

    return image, res

parser = argparse.ArgumentParser()
parser.add_argument('--images', default='./Data_folder/MR', help='path to the images')
parser.add_argument('--labels', default='./Data_folder/MR_label', help='path to the labels')
parser.add_argument('--split_val', default=8, help='number of images for validation')
parser.add_argument('--split_test', default=7, help='number of images for testing')
args = parser.parse_args()

if __name__ == "__main__":

    list_images = lstFiles(args.images)
    list_labels = lstFiles(args.labels)

    # mapIndexPosition = list(zip(list_images, list_labels))  # shuffle order list
    # random.shuffle(mapIndexPosition)
    # list_images, list_labels = zip(*mapIndexPosition)

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

        image, label = uniform_img_dimensions(image, label)

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

        image, label = uniform_img_dimensions(image, label)

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

        image, label = uniform_img_dimensions(image, label)

        image_directory = os.path.join('./Data_folder/images/test', f"image{i:d}.nii")
        label_directory = os.path.join('./Data_folder/labels/test', f"label{i:d}.nii")

        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

