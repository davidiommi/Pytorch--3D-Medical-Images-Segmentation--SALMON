![Salmon-logo-1](images/salmon.JPG)
# SALMON: Segmentation deep learning ALgorithm based on MONai toolbox
SALMON is a computational toolbox for segmentation using neural networks.
SALMON is based on MONAI: a PyTorch-based, open-source framework for deep learning in healthcare imaging. (https://github.com/Project-MONAI/MONAI)

### Requirements
See requirements.txt list

### Python scripts and their function

- organize_folder_structure.py: Organize the data in the folder structure for the network

- init.py: List of options used to train the network. 

- check_loader_patches: Shows example of patches fed to the network during the training  

- networks.py: the architectures available for segmentation.

- train.py: Runs the training

- predict_single_image.py: It launches the inference on a single input image chosen by the user.

## Usage

1) Use first organize_folder_structure.py to create organize the data in the following folder structure:

    .
	├── Data_folder
	|   ├── images              
	|   |   ├── train                       
	|   |   |   ├── image1.nii                                  
	|   |   |   └── image2.nii                       
	|   |   └── val             
	|   |   |   ├── image3.nii                                              
	|   |   |   └── image4.nii    
	|   ├── labels              
	|   |   ├── train                       
	|   |   |   ├── label1.nii                                  
	|   |   |   └── label2.nii                       
	|   |   └── val             
	|   |   |   ├── label3.nii                                              
	|   |   |   └── label4.nii
	
    .
	├── Data_folder                   
	|   ├── train_set              
	|   |   ├── patient_1             # Training
	|   |   |   ├── image.nii              
	|   |   |   └── label.nii                     
	|   |   └── patient_2             
	|   |   |   ├── image.nii              
	|   |   |   └── label.nii              
	|   ├── test_set               
	|   |   ├── patient_3             # Testing
	|   |   |   ├── image.nii              
	|   |   |   └── label.nii              
	|   |   └── patient_4             
	|   |   |   ├── image.nii              
	|   |   |   └── label.nii        

2) Modify the init.py to set the parameters and start the training/testing on the data.
Afterwards launch the train.py for training. 	