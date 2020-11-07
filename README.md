# Wide_Residual_Unet's main features


# SEnet
<hr>
Resblock include SEnet. It connects not deep wide but better performance NN for segmentation and generation

![SENet](https://user-images.githubusercontent.com/48679574/98444753-c7ec7a80-2156-11eb-909d-a7e8caa784bc.png)







# image preprocess(for imprping accuracy)
<hr>
I use "Histogram averaging" mainly just like bellow


<b>【HH Histogram averagings】 & 【HV Histogram averaging】</b>
<hr>

<img src="https://user-images.githubusercontent.com/48679574/98444008-8659d080-2152-11eb-9c47-07feccc88cee.png" width="500px">




Details of these code and logics I've done were written below site.(my blog)

https://trafalbad.hatenadiary.jp/entry/2019/12/01/170905







# generate image by WSEUnet
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

