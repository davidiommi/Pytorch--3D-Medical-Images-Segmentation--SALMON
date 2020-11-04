# SALMON: Segmentation of Thoracic aorta from CT

No aortic arc is available. Example images:
Left: CT | Center: segmentation result | Right: Original label

![Image](1.gif)![result](2.gif)![label](3.gif)
*******************************************************************************
## Requirements
See requirements.txt list. We use "nvcr.io/nvidia/pytorch:19.03-py3" docker and we install all the requirements with:
"pip install -r requirements.txt"
*******************************************************************************

### Weights link for download
https://drive.google.com/file/d/1RrA17IGNX_vH6FirJezuNzWGZYYkaC1G/view?usp=sharing
	
### Inference:
Launch "predict_single_image.py" to test the network. Modify the parameters in the parse section to select the path of the weights, images to infer and result. 

```console
python predict_single_image.py --image './Data_folder/images/train/image13.nii' --label './Data_folder/labels/train/label13.nii' --result './Data_folder/results/train/prova.nii' --weights './best_metric_model.pth'
```
*******************************************************************************
