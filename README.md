![Salmon-logo-1](images/salmon.JPG)
# SALMON v.2: Segmentation deep learning ALgorithm based on MONai toolbox
- SALMON is a computational toolbox for segmentation using neural networks (3D patches-based segmentation)
- SALMON is based on  NN-UNET and MONAI: PyTorch-based, open-source frameworks for deep learning in healthcare imaging. 
(https://github.com/Project-MONAI/MONAI)
(https://github.com/MIC-DKFZ/nnUNet)

This is my "open-box" version if I want to modify the parameters for some particular task, while the two above are hard-coded.

*******************************************************************************
## Requirements
We download the official MONAI DockerHub, with the latest MONAI version. Please visit https://docs.monai.io/en/latest/installation.html
Additional packages can be installed with "pip install -r requirements.txt"
*******************************************************************************
## Python scripts and their function

- organize_folder_structure.py: Organize the data in the folder structure (training,validation,testing) for the network. Labels are resampled and resized to the corresponding image, to avoid conflicts.

- init.py: List of options used to train the network. 

- check_loader_patches: Shows example of patches fed to the network during the training.  

- networks.py: The architecture available for segmentation is a nn-Unet.

- train.py: Runs the training

- predict_single_image.py: It launches the inference on a single input image chosen by the user.
*******************************************************************************
## Usage
### Folders structure:

Use first "organize_folder_structure.py" to create organize the data.
Modify the input parameters to select the two folders: images and labels folders with the dataset.

    .
	├── Data_folder                   
	|   ├── CT               
	|   |   ├── 1.nii 
    |   |   ├── 2.nii 	
	|   |   └── 3.nii                     
	|   ├── CT_labels                         
	|   |   ├── 1.nii 
    |   |   ├── 2.nii 	
	|   |   └── 3.nii  

Data structure after running it:

	.
	├── Data_folder                   
	|   ├── images              
	|   |   ├── train             
	|   |   |   ├── image1.nii              
	|   |   |   └── image2.nii                     
	|   |   └── val             
	|   |   |   ├── image3.nii             
	|   |   |   └── image4.nii
	|   |   └── test             
	|   |   |   ├── image5.nii              
	|   |   |   └── image6.nii
	|   ├── labels              
	|   |   ├── train             
	|   |   |   ├── label1.nii              
	|   |   |   └── label2.nii                     
	|   |   └── val             
	|   |   |   ├── label3.nii             
	|   |   |   └── label4.nii
	|   |   └── test             
	|   |   |   ├── label5.nii              
	|   |   |   └── label6.nii
 
*******************************************************************************
### Training:
- Modify the "init.py" to set the parameters and start the training/testing on the data. Read the descriptions for each parameter.
- Afterwards launch the train.py for training. Tensorboard is available to monitor the training:	

![training](images/salmon3.JPG)![training2](images/salmon4.JPG)

Sample images: on the left side the image, in the middle the result of the segmentation and on the right side the true label
The following images show the segmentation of carotid artery from MR sequence

![Image](images/image.gif)![result](images/result.gif)![label](images/label.gif)
*******************************************************************************
### Inference:
Launch "predict_single_image.py" to test the network. Modify the parameters in the parse section to select the path of the weights, images to infer and result. 
*******************************************************************************
### Tips:
- Use and modify "check_loader_patches.py" to check the patches fed during training. 
- "Organize_folder_structure.py" solves dimensionality conflicts if the label has different size and resolution of the image. Check on 3DSlicer or ITKSnap if your data are correctly centered-overlaid.
- The "networks.py" calls the nn-Unet, which adapts itself to the input data (resolution and patches size). The script also saves the graph of you network, so you can visualize it. 
- Is it possible to add other networks, but for segmentation the U-net architecture is the state of the art.
- During the training phase, the script crops the image background and pad the image if the post-cropping size is smaller than the patch size you set in init.py


### Sample script inference
- The label can be omitted (None) if you segment an unknown image. You can add the --resolution if you resampled the data during training (look at the argsparse in the code).
```console
python predict_single_image.py --image './Data_folder/images/train/image13.nii' --label './Data_folder/labels/train/label13.nii' --result './Data_folder/results/train/prova.nii' --weights './best_metric_model.pth'
```
*******************************************************************************
### Multi-channel segmentation: 

The subfolder "multi_label_segmentation_example" shows how to change the code for multi_label scenario.
The example segment the prostate (1 channel input) in the transition zone and peripheral zone (2 channels output).

Some note:
- You must add an additional channel for the background. Example: 0 background, 1 prostate, 2 prostate tumor = 3 out channels in total.
- Tensorboard can show you all segmented channels, but for now the metric is the Mean-Dice (of all channels). If you want to evaluate the Dice score for each channel you 
  have to modify a bit the plot_dice function. I will do it...one day...who knows...maybe not
- The loss is the DiceLoss + CrossEntropy. You can modify it if you want to try others (https://docs.monai.io/en/latest/losses.html#diceloss)

Check more examples at https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb.
