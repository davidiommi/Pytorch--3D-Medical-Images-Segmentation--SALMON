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
from monai.transforms import (Compose, LoadNiftid, AddChanneld, Transpose,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined,
    Spacingd, Orientationd, CropForegroundd, RandZoomd, RandShiftIntensityd, RandGaussianNoised, BorderPadd,RandAdjustContrastd, NormalizeIntensityd,RandFlipd, ScaleIntensityRanged)


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

        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys=['image']),
        ScaleIntensityd(keys=['image']),
        # Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),
        # RandFlipd(keys=['image', 'label'], prob=1, spatial_axis=2),
        # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1,
        #             rotate_range=(np.pi / 36, np.pi / 4, np.pi / 36)),
        # Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1,
        #                sigma_range=(5, 8), magnitude_range=(100, 200), scale_range=(0.20, 0.20, 0.20)),
        # RandAdjustContrastd(keys=['image'], gamma=(0.5, 3), prob=1),
        # RandGaussianNoised(keys=['image'], prob=1, mean=np.random.uniform(0, 0.5), std=np.random.uniform(0, 1)),
        # RandShiftIntensityd(keys=['image'], offsets=np.random.uniform(0,0.3), prob=1),
        # BorderPadd(keys=['image', 'label'],spatial_border=(16,16,0)),
        RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
        # Orientationd(keys=["image", "label"], axcodes="PLI"),
        ToTensord(keys=['image', 'label'])
    ]

    transform = Compose(monai_transforms)

    check_ds = monai.data.Dataset(data=data_dicts, transform=transform)

    loader = DataLoader(check_ds, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(loader)
    im, seg = (check_data['image'][0], check_data['label'][0])
    print(im.shape, seg.shape)

    vol = im[0].numpy()
    mask = seg[0].numpy()

    print(vol.shape)
    plot3d(vol)
    plot3d(mask)
