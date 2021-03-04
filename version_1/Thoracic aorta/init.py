import argparse
import os

class Options():
    """This class defines options used during both training and test time."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):

        # basic parameters
        parser.add_argument('--images_folder', type=str, default='./Data_folder/images')
        parser.add_argument('--labels_folder', type=str, default='./Data_folder/labels')
        parser.add_argument('--increase_factor_data',  default=4, help='Increase data number per epoch')
        parser.add_argument('--preload', type=str, default='best_metric_model.pth')
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')

        # dataset parameters
        parser.add_argument('--patch_size', default=(128, 128, 128), help='Size of the patches extracted from the image')
        parser.add_argument('--resolution', default=(1.3671875 * 1.2, 1.3671875 * 1.2, 3.0 * 1.2), help='New Resolution')
        parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        parser.add_argument('--in_channels', default=1, type=int, help='Channels of the input')
        parser.add_argument('--out_channels', default=1, type=int, help='Channels of the output')

        # training parameters
        parser.add_argument('--epochs', default=300, help='Number of epochs')
        parser.add_argument('--net', type=str, default='Unet_Monai', help='[HighResNet | Unet_Monai | '
                                                                          'Unet_David]')
        parser.add_argument('--lr', default=0.01, help='Learning rate')
        parser.add_argument('--weight_decay', default=1e-4, help='patience')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='lr rate: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters')

        # Inference
        # This is just a trick to make the predict script working
        parser.add_argument('--result', default=None, help='Keep this empty and go to predict_single_image script')
        parser.add_argument('--weights', default=None, help='Keep this empty and go to predict_single_image script')

        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()
        # set gpu ids
        if opt.gpu_ids != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        return opt





