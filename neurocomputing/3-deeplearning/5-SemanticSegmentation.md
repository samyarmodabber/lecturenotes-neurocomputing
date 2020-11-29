# Semantic segmentation

Slides: [pdf](https://www.tu-chemnitz.de/informatik/KI/edu/neurocomputing/lectures/pdf/3.5-SegNet.pdf)


## Semantic segmentation

**Semantic segmentation** is a class of segmentation methods where you use knowledge about the identity of objects to partition the image pixel-per-pixel.

```{figure} ../img/semanticsegmentation-example.png
---
width: 100%
---
Semantic segmentation. Source : <https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef>
```

Classical segmentation methods only rely on the similarity between neighboring pixels, they do not use class information. The output of a semantic segmentation is another image, where each pixel represents the class.

The classes can be binary, for example foreground/background, person/not, etc. Semantic segmentation networks are used for example in Youtube stories to add **virtual backgrounds** (background matting).

```{figure} ../img/segmentation-virtualbackground.gif
---
width: 40%
---
Virtual backgrounds. Source: <https://ai.googleblog.com/2018/03/mobile-real-time-video-segmentation.html>
```


Clothes can be segmented to allow for virtual try-ons.

```{figure} ../img/semanticsegmentation-clothes.png
---
width: 100%
---
Virtual try-ons. Source: {cite}`Wang2018b`.
```


There are many datasets freely available, but annotating such data is very painful, expensive and error-prone.

* PASCAL VOC 2012 Segmentation Competition
* COCO 2018 Stuff Segmentation Task
* BDD100K: A Large-scale Diverse Driving Video Database
* Cambridge-driving Labeled Video Database (CamVid)
* Cityscapes Dataset
* Mapillary Vistas Dataset
* ApolloScape Scene Parsing
* KITTI pixel-level semantic segmentation

```{figure} ../img/kitti-example.png
---
width: 100%
---
Semantic segmentation on the KITTI dataset. Source: <http://www.cvlibs.net/datasets/kitti/>
```

### Output encoding

Each pixel of the input image is associated to a label (as in classification).

```{figure} ../img/segnet-encoding.png
---
width: 100%
---
Semantic labels. Source : <https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef>
```

A **one-hot encoding** of the segmented image is therefore a tensor:

```{figure} ../img/segnet-encoding2.png
---
width: 100%
---
One-hot encoding of semantic labels. Source : <https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef>
```

### Fully convolutional networks

A **fully convolutional network** only has convolutional layers and learns to predict the output tensor. The last layer has a pixel-wise softmax activation. We minimize the **pixel-wise cross-entropy loss**

$$\mathcal{L}(\theta) = \mathbb{E}_\mathcal{D} [- \sum_\text{pixels} \sum_\text{classes} t_i \, \log y_i]$$

```{figure} ../img/fullyconvolutional.png
---
width: 100%
---
Fully convolutional network. Source : <http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf>
```

Downside: the image size is preserved throughout the network: computationally expensive. It is therefore difficult to increase the number of features in each convolutional layer.


## SegNet: segmentation network

**SegNet** {cite}`Badrinarayanan2016` has an **encoder-decoder** architecture, with max-pooling to decrease the spatial resolution while increasing the number of features. But what is the inverse of max-pooling? **Upsampling operation**.

```{figure} ../img/segnet.png
---
width: 100%
---
**SegNet** {cite}`Badrinarayanan2016`.
```

Nearest neighbor and Bed of nails would just make random decisions for the upsampling. In SegNet, max-unpooling uses the information of the corresponding max-pooling layer in the encoder to place pixels adequately.


```{figure} ../img/max-pooling-inverse.png
---
width: 100%
---
Upsampling methods. Source : <http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf>
```

Another popular option in the followers of SegNet is the **transposed convolution**. The original feature map is upsampled by putting zeros between the values and a learned filter performs a regular convolution to produce an upsampled feature map. This works well when convolutions with stride are used in the encoder, but it is quite expensive computationally.

```{figure} ../img/padding_strides_transposed.gif
---
width: 50%
---
Transposed convolution. Source : <https://github.com/vdumoulin/conv_arithmetic> 
```

```{figure} ../img/transposedconvolution.png
---
width: 70%
---
Transposed convolution. Source : <http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf> 
```

## U-Net


The problem of SegNet is that small details (small scales) are lost because of the max-pooling. the segmentation is not precise. The solution proposed by **U-Net** {cite}`Ronneberger2015` is to add **skip connections** (as in ResNet) between different levels of the encoder-decoder. The final segmentation depends both on:

* large-scale information computed in the middle of the encoder-decoder.
* small-scale information processed in the early layers of the encoder.

```{figure} ../img/unet.png
---
width: 100%
---
U-Net {cite}`Ronneberger2015`.
```


## Mask R-CNN

For many applications, segmenting the background is useless. A two-stage approach can save computations. **Mask R-CNN** {cite}`He2018` uses faster R-CNN to extract bounding boxes around interesting objects, followed by the prediction of a **mask** to segment the object.

```{figure} ../img/mask-r-cnn.png
---
width: 100%
---
Mask R-CNN {cite}`He2018`.
```

```{figure} ../img/mask-r-cnn-result.png
---
width: 100%
---
Mask R-CNN {cite}`He2018`.
```

<div class='embed-container'><iframe src='https://www.youtube.com/embed/OOT3UIXZztE' frameborder='0' allowfullscreen></iframe></div>
