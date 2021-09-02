# Wide Residual block Unet with onnx

# Version
- python 3.7.0
- tensorflow=='2.3.0'
- keras=='2.3.1'
- onnx==1.10.1
- keras2onnx=='1.9.0'
- onnxruntime=='1.8.1'


# SEnet
<hr>
Resblock include SEnet. It connects not deep wide but better performance NN for segmentation and generation

![SENet](https://user-images.githubusercontent.com/48679574/98444753-c7ec7a80-2156-11eb-909d-a7e8caa784bc.png)







# Unet types
- Wide SEblock Unet
- EfficientNet Unet
- Wide SEblock Unet like ResNet-RS with mish activation

# Onnx convert

you can convert all models to onnx.
```
cd onnx
python3 onnx_convert.py
```


# generate image 
<hr>
Unet with residual blocks for generating similar images


## loss curve

![loss_curve](https://user-images.githubusercontent.com/48679574/98444055-e3ee1d00-2152-11eb-9ad4-a75bd3659177.png)



## Input image

<img src="https://user-images.githubusercontent.com/48679574/98444065-f8321a00-2152-11eb-934d-3f14722065f2.png" width="400px">


## generate image1

<img src="https://user-images.githubusercontent.com/48679574/98444075-0b44ea00-2153-11eb-8398-7762cff25291.png" width="400px">


## generate iamge2 

<img src="https://user-images.githubusercontent.com/48679574/98444082-17c94280-2153-11eb-8ddd-727d217fdfab.png" width="400px">

