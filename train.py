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


from init import Options
from networks import *
import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_3d, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (Compose, LoadNiftid, AddChanneld, Transpose,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined,
    Spacingd, Orientationd, RandGaussianNoised, RandAdjustContrastd,NormalizeIntensityd,RandFlipd)

from monai.visualize import plot_2d_or_3d_image
from monai.engines import get_devices_spec

def main():
    opt = Options().parse()
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if opt.gpu_ids != '-1':
        num_gpus = len(opt.gpu_ids.split(','))
    else:
        num_gpus = 0
    print('number of GPU:', num_gpus)

    # Data loader creation

    # train images
    train_images = sorted(glob(os.path.join(opt.images_folder, 'train', 'image*.nii')))
    train_segs = sorted(glob(os.path.join(opt.labels_folder, 'train', 'label*.nii')))

    # validation images
    val_images = sorted(glob(os.path.join(opt.images_folder, 'val', 'image*.nii')))
    val_segs = sorted(glob(os.path.join(opt.labels_folder, 'val', 'label*.nii')))

    # augment the data list for training
    for i in range(int(opt.increase_factor_data)):

        train_images.extend(train_images)
        train_segs.extend(train_segs)

        val_images.extend(val_images)
        val_segs.extend(val_segs)

    print('Number of training patches per epoch:', len(train_images))
    print('Number of validation patches per epoch:', len(val_images))

    # Creation of data directories for data_loader

    train_dicts = [{'image': image_name, 'label': label_name}
                  for image_name, label_name in zip(train_images, train_segs)]

    val_dicts = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(val_images, val_segs)]

    if opt.resolution is not None:
        train_transforms = [
            LoadNiftid(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            NormalizeIntensityd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 36, np.pi * 4)),
            Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                           sigma_range=(5, 8), magnitude_range=(100, 200)),
            RandAdjustContrastd(keys=['image'], prob=0.1),
            RandGaussianNoised(keys=['image'],
                               prob=0.1, mean=0.0, std=0.1),
            RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
        ]

        val_transforms = [
            LoadNiftid(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            NormalizeIntensityd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),
            RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
        ]
    else:
        train_transforms = [
            LoadNiftid(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            NormalizeIntensityd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 36, np.pi * 4)),
            Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                           sigma_range=(5, 8), magnitude_range=(100, 200)),
            RandAdjustContrastd(keys=['image'], prob=0.1),
            RandGaussianNoised(keys=['image'],
                               prob=0.1, mean=0.0, std=0.1),
            RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
        ]

        val_transforms = [
            LoadNiftid(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            NormalizeIntensityd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
        ]

    train_transforms = Compose(train_transforms)
    val_transforms = Compose(val_transforms)

    # create a training data loader
    check_train = monai.data.Dataset(data=train_dicts, transform=train_transforms)
    train_loader = DataLoader(check_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    check_val = monai.data.Dataset(data=val_dicts, transform=val_transforms)
    val_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, pin_memory=torch.cuda.is_available())

    # try to use all the available GPUs
    devices = get_devices_spec(None)

    # build the network
    net = build_net()
    net.cuda()

    if num_gpus > 1:
        net = torch.nn.DataParallel(net)

    if opt.preload is not None:
        net.load_state_dict(torch.load(opt.preload))

    dice_metric = DiceMetric(include_background=True, to_onehot_y=False, sigmoid=True, reduction="mean")
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    lr = opt.lr
    optim = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=True)
    net_scheduler = get_scheduler(optim, opt)

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(opt.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{opt.epochs}")
        net.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].cuda(), batch_data["label"].cuda()
            optim.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            epoch_len = len(check_train) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        update_learning_rate(net_scheduler, optim)

        if (epoch + 1) % val_interval == 0:
            net.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["image"].cuda(), val_data["label"].cuda()
                    roi_size = opt.patch_size
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    metric_count += len(value)
                    metric_sum += value.item() * len(value)
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(net.state_dict(), "best_metric_model.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                val_outputs = (val_outputs.sigmoid() >= 0.5).float()
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()