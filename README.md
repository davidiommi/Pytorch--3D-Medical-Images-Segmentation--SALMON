![Salmon-logo-1](images/salmon.JPG)
# SALMON v.2: Segmentation deep learning ALgorithm based on MONai toolbox
- SALMON is a computational toolbox for segmentation using neural networks (3D patches-based segmentation)
- SALMON is based on MONAI 0.7.0 : PyTorch-based, open-source frameworks for deep learning in healthcare imaging. 
(https://github.com/Project-MONAI/MONAI)
(https://github.com/MIC-DKFZ/nnUNet)
(https://arxiv.org/abs/2103.10504)

This is my "open-box" version if I want to modify the parameters for some particular task, while the two above are hard-coded. The monai 0.5.0 folder contains the previous versions based on the old monai version.

*******************************************************************************
## Requirements
Follow the steps in "installation_commands.txt". Installation via Anaconda and creation of a virtual env to download the python libraries and pytorch/cuda.
*******************************************************************************
## Python scripts and their function

- organize_folder_structure.py: Organize the data in the folder structure (training,validation,testing) for the network. 
Labels are resampled and resized to the corresponding image, to avoid array size conflicts. You can set here a new image resolution for the dataset. 

- init.py: List of options used to train the network. 

- check_loader_patches: Shows example of patches fed to the network during the training.  

- networks.py: The architectures available for segmentation are nn-Unet and UneTR (based on Visual transformers)

- train.py: Runs the training

- predict_single_image.py: It launches the inference on a single input image chosen by the user.
*******************************************************************************
## Usage
### Folders structure:

Use first "organize_folder_structure.py" to create organize the data.
Modify the input parameters to select the two folders: images and labels folders with the dataset. Set the resolution of the images here before training.

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
	|   ├── CT  
	|   ├── CT_labels 
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
- Afterwards launch the "train.py" for training. Tensorboard is available to monitor the training ("runs" folder created)	
- Check and modify the train_transforms applied to the images  in "train.py" for your specific case. (e.g. In the last update there is a HU windowing for CT images)

Sample images: the following images show the segmentation of carotid artery from MRI sequence

![Image](images/image.gif)![result](images/result.gif)

Sample images: the following images show the multi-label segmentation of prostate transition zone and peripheral zone from MRI sequence

![Image1](images/prostate.gif)![result1](images/prostate_inf.gif)!

*******************************************************************************
### Inference:
- Launch "predict_single_image.py" to test the network. Modify the parameters in the parse section to select the path of the weights, images to infer and result. 
- You can test the model on a new image, with different size and resolution from the training. The script will resample it before the inference and give you a mask
with same size and resolution of the source image.
*******************************************************************************
### Tips:
- Use and modify "check_loader_patches.py" to check the patches fed during training. 
- The "networks.py" calls the nn-Unet, which adapts itself to the input data (resolution and patches size). The script also saves the graph of you network, so you can visualize it. 
- "networks.py" includes also UneTR (based on Visual transformers). This is experimental. For more info check (https://arxiv.org/abs/2103.10504) and https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unetr_btcv_segmentation_3d.ipynb
- Is it possible to add other networks, but for segmentation the U-net architecture is the state of the art.

### Sample script inference
- The label can be omitted (None) if you segment an unknown image. You have to add the --resolution if you resampled the data during training (look at the argsparse in the code).
```console
python predict_single_image.py --image './Data_folder/image.nii' --label './Data_folder/label.nii' --result './Data_folder/prova.nii' --weights './best_metric_model.pth'
```
*******************************************************************************

### Some note:
- Tensorboard can show you all segmented channels, but for now the metric is the Mean-Dice (of all channels). If you want to evaluate the Dice score for each channel you 
  have to modify a bit the plot_dice function. I will do it...one day...who knows...maybe not
- The loss is the DiceLoss + CrossEntropy. You can modify it if you want to try others (https://docs.monai.io/en/latest/losses.html#diceloss)

Check more examples at https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/.

### UneTR Notes from the authors:

Feature_size and pos_embed are the parameters that need to changed to adopt it for your application of interest. Other parameters that are mentioned come from Vision Transformer (ViT) default hyper-parameters (original architecture). In addition, the new revision of UNETR paper with more descriptions is now publicly available. Please check for more details:
https://arxiv.org/pdf/2103.10504.pdf

Now let's look at each of these hyper-parameters in the order of importance:

- feature_size : In UNETR, we multiply the size of the CNN-based features in the decoder by a factor of 2 at every resolution ( just like the original UNet paper). By default, we set this value to 16 ( to make the entire network lighter). However using larger values such as 32 can improve the segmentation performance if GPU memory is not an issue. Figure2 of the paper also shows this in details.

- pos_embed: this determines how the image is divided into non-overlapping patches. Essentially, there are 2 ways to achieve this ( by setting it to conv or perceptron). Let's further dive into it for more information:
First is by directly applying a convolutional layer with the same stride and kernel size of the patch size and with feature size of the hidden size in the ViT model. Second is by first breaking the image into patches by properly resizing the tensor ( for which we use einops) and then feed it into a perceptron (linear) layer with a hidden size of the ViT model. Our experiments show that for certain applications such as brain segmentation with multiple modalities (e.g. 4 modes such as T1,T2 etc.), using the convolutional layer works better as it takes into account all modes concurrently. For CT images ( e.g. BTCV multi-organ segmentation), we did not see any difference in terms of performance between these two approaches.

- hidden_size : this is the size of the hidden layers in the ViT encoder. We follow the original ViT model and set this value to 768. In addition, the hidden size should be    divisible by the number of attention heads in the ViT model.

- num_heads : in the multi-headed self-attention block, this is the number of attention heads. Following the ViT architecture, we set it to 12.

- mlp_dim : this is the dimension of the multi-layer perceptrons (MLP) in the transformer encoder. Again, we follow the ViT model and set this to 3072 as default value to be     consistent with their architecture.

