![Salmon-logo-1](images/salmon.JPG)
# SALMON v.2: Segmentation deep learning ALgorithm based on MONai toolbox
SALMON is a computational toolbox for segmentation using neural networks (3D patches-based segmentation)
SALMON is based on MONAI: a PyTorch-based, open-source framework for deep learning in healthcare imaging. (https://github.com/Project-MONAI/MONAI)
*******************************************************************************
## Requirements
We download the official MONAI DockerHub, with the latest MONAI version. Please visit https://docs.monai.io/en/latest/installation.html
Additional packages can be installed with "pip install -r requirements.txt"
*******************************************************************************
## Python scripts and their function

- organize_folder_structure.py: Organize the data in the folder structure (training,validation,testing) for the network

- init.py: List of options used to train the network. 

- check_loader_patches: Shows example of patches fed to the network during the training  

- networks.py: The architecture available for segmentation is a nn-Unet.

- train.py: Runs the training

- predict_single_image.py: It launches the inference on a single input image chosen by the user.
*******************************************************************************
## Usage
### Folders structure:
Use first "organize_folder_structure.py" to create organize the data in the following folder structure:


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

Modify the input parameters to select folders and divide the dataset 
*******************************************************************************
### Training:
Modify the "init.py" to set the parameters and start the training/testing on the data. Read the descriptions for each parameter.
Afterwards launch the train.py for training. Tensorboard is available to monitor the training:	

![training](images/salmon3.JPG)![training2](images/salmon4.JPG)![training3](images/salmon5.JPG)![training2´4](images/salmon6.JPG)

Sample images: on the left side the image, in the middle the result of the segmentation and on the right side the true label
The following images show the segmentation of carotid artery from MR sequence

![Image](images/image.gif)![result](images/result.gif)![label](images/label.gif)
*******************************************************************************
### Inference:
Launch "predict_single_image.py" to test the network. Modify the parameters in the parse section to select the path of the weights, images to infer and result. 
*******************************************************************************
### Tips:
Use and modify "check_loader_patches.py" to check the patches fed during training. 
The "networks.py" calls the nn-Unet, which adapts itself to the input data (resolution and patches size) 
Is it possible to add other networks, but for segmentation the U-net architecture is the state of the art.


### Sample script inference
```console
python predict_single_image.py --image './Data_folder/images/train/image13.nii' --label './Data_folder/labels/train/label13.nii' --result './Data_folder/results/train/prova.nii' --weights './best_metric_model.pth'
```
*******************************************************************************
### Multi-channel segmentation: 

To implement the multilabel segmentation a few lines must be added:
- In the transforms section of the data channels must added and concatenated
- The loss function must be modified (Dice to softDice)

Check the example at https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb as example.
