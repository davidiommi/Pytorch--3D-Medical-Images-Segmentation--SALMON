# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from glob import glob
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from init import Options
import monai
from monai.data import ArrayDataset, GridPatchDataset, create_test_image_3d
from monai.transforms import (Compose, LoadImaged, AddChanneld, Transpose, Resized, CropForegroundd, CastToTyped,RandGaussianSmoothd,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, SpatialPadd,
    Spacingd, Orientationd, RandZoomd, ThresholdIntensityd, RandShiftIntensityd, RandGaussianNoised, BorderPadd,RandAdjustContrastd, NormalizeIntensityd,RandFlipd, ScaleIntensityRanged)


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap= 'gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def plot3d(image):
    original=image
    original = np.rot90(original, k=-1)
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, original)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


if __name__ == "__main__":

    opt = Options().parse()

    train_images = sorted(glob(os.path.join(opt.images_folder, 'train', 'image*.nii')))
    train_segs = sorted(glob(os.path.join(opt.labels_folder, 'train', 'label*.nii')))

    data_dicts = [{'image': image_name, 'label': label_name}
                  for image_name, label_name in zip(train_images, train_segs)]

    monai_transforms = [

        LoadImaged(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
        ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
        CropForegroundd(keys=['image', 'label'], source_key='image', start_coord_key='foreground_start_coord',
                        end_coord_key='foreground_end_coord', ),  # crop CropForeground
        NormalizeIntensityd(keys=['image']),
        ScaleIntensityd(keys=['image']),
        # Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),

        SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method= 'end'),
        RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
        ToTensord(keys=['image', 'label','foreground_start_coord', 'foreground_end_coord'],)
    ]

    transform = Compose(monai_transforms)
    check_ds = monai.data.Dataset(data=data_dicts, transform=transform)
    loader = DataLoader(check_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(loader)
    im, seg, coord1, coord2 = (check_data['image'][0], check_data['label'][0],check_data['foreground_start_coord'][0],
                      check_data['foreground_end_coord'][0])

    print(im.shape, seg.shape, coord1, coord2)

    vol = im[0].numpy()
    mask = seg[0].numpy()

    print(vol.shape, mask.shape)
    plot3d(vol)
    plot3d(mask)
