# Generative adversarial networks

Slides: [pdf](https://www.tu-chemnitz.de/informatik/KI/edu/neurocomputing/lectures/pdf/3.8-GAN.pdf)

## Generative adversarial networks

<div class='embed-container'><iframe src='https://www.youtube.com/embed/CWgvnu8Qtug' frameborder='0' allowfullscreen></iframe></div>

### Generative models

An autoencoder learns to first encode inputs in a **latent space** and then use a generative model to model the data distribution.

$$\mathcal{L}_\text{autoencoder}(\theta, \phi) = \mathbb{E}_{\mathbf{x} \in \mathcal{D}, \mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})} [ - \log p_\theta(\mathbf{z})]$$

Couldn't we learn a decoder using random noise as input but still learning the distribution of the data?

$$\mathcal{L}_\text{GAN}(\theta, \phi) = \mathbb{E}_{\mathbf{z} \sim \mathcal{N}(0, 1)} [ - \log p_\theta(\mathbf{z}) ]$$

After all, this is how random numbers are generated: a uniform distribution of pseudo-random numbers is transformed into samples of another distribution using a mathematical formula.

```{figure} ../img/generation-distribution.jpeg
---
width: 100%
---
Random numbers are generated using a standard distribution as source of randomness. Source: <https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29>
```

The problem is how to estimate the discrepancy between the true distribution and the generated distribution when we only have samples. The Maximum Mean Discrepancy (MMD) approach allows to do that, but does not work very well in highly-dimensional spaces.


```{figure} ../img/gan-principle2.png
---
width: 100%
---
The generative sample should learn to minimize the statistical distance between the true distribution and the parameterized distribution using samples. Source: <https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29>
```

### Architecture of a GAN

The **Generative Adversarial Network** (GAN, {cite}`Goodfellow2014`) is a smart way of providing a loss function to the generative model. It is composed of two parts:

* The **Generator** (or decoder) produces an image based on latent variables sampled from some random distribution (e.g. uniform or normal).
* The **Discriminator** has to recognize real images from generated ones.


```{figure} ../img/gan-simple2.png
---
width: 100%
---
Architecture of a GAN. The generator only sees noisy latent representations and outputs a reconstruction. The discriminator gets alternatively real or generated inputs and predicts whether it is real or fake. Source: <https://www.oreilly.com/library/view/java-deep-learning/9781788997454/60579068-af4b-4bbf-83f1-e988fbe3b226.xhtml>
```

The generator and the discriminator are in competition with each other. The discriminator uses pure **supervised learning**: we know if the input is real or generated (binary classification) and train the discriminator accordingly. The generator tries to fool the discriminator, without ever seeing the data!

```{figure} ../img/gan-principle.png
---
width: 100%
---
Principle of a GAN. Source: <https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29>
```

### GAN loss

Let's define $x \sim P_\text{data}(x)$ as a real image from the dataset and $G(z)$ as an image generated by the generator,  where $z \sim P_z(z)$ is a random input. The output of the discriminator is a single sigmoid neuron:

* $D(x) = 1$ for real images.
* $D(G(z)) = 0$ for generated images

The discriminator wants both $D(x)$ and $1-D(G(z))$ to be close from 1, so the goal of the discriminator is to **minimize** the negative log-likelihood (cross-entropy) of classifying correctly the data:

$$
    \mathcal{L}(D) = \mathbb{E}_{x \sim P_\text{data}(x)} [ - \log D(x)] + \mathbb{E}_{z \sim P_z(z)} [ - \log(1 - D(G(z)))]
$$

It is similar to logistic regression: $x$ belongs to the positive class, $G(z)$ to the negative class.

The goal of the generator is to **maximize** the negative log-likelihood of the discriminator being correct on the generated images, i.e. fool it: 

$$
    \mathcal{J}(G) = \mathbb{E}_{z \sim P_z(z)} [ - \log(1 - D(G(z)))]
$$

The generator tries to maximize what the discriminator tries to minimize.

Putting both objectives together, we obtain the following **minimax** problem:

$$
    \min_G \max_D \, \mathcal{V}(D, G) = \mathbb{E}_{x \sim P_\text{data}(x)} [\log D(x)] + \mathbb{E}_{z \sim P_z(z)} [\log(1 - D(G(z)))]
$$

$D$ and $G$ compete on the same objective function: one tries to maximize it, the other to minimize it. Note that the generator $G$ never sees the data $x$: all it gets is a **backpropagated gradient** through the discriminator:

$$\nabla_{G(z)} \, \mathcal{V}(D, G) = \nabla_{D(G(z))} \, \mathcal{V}(D, G) \times \nabla_{G(z)} \, D(G(z))$$

It informs the generator which **pixels** are the most responsible for an eventual bad decision of the discriminator.

This objective function can be optimized when the generator uses gradient descent and the discriminator gradient ascent: just apply a minus sign on the weight updates!

$$
    \min_G \max_D V(D, G) = \mathbb{E}_{x \sim P_\text{data}(x)} [\log D(x)] + \mathbb{E}_{z \sim P_z(z)} [\log(1 - D(G(z)))]
$$

Both can therefore use the usual **backpropagation** algorithm to adapt their parameters. The discriminator and the generator should reach a **Nash equilibrium**: they try to beat each other, but both become better over time.


```{figure} ../img/gan-loss.png
---
width: 100%
---
The generator and discriminator loss functions reach an equilibrium, it is quite hard to tell when the network has converged. Source: Research project - Vivek Bakul Maru - TU Chemnitz
```

### Variants

DCGAN {cite}`Radford2015` is the convolutional version of GAN, using transposed convolutions in the generator and concolutions with stride in the discriminator.

```{figure} ../img/dcgan-flat.png
---
width: 100%
---
DCGAN {cite}`Radford2015`.
```

```{figure} ../img/dcgan.png
---
width: 100%
---
Results of DCGAN {cite}`Radford2015`.
```

GAN are quite sensible to train: the discriminator should not become too good too early, otherwise there is no usable gradient for the generator. In practice, one updates the generator more often than the discriminator. There has been many improvements on GANs to stabilizes training (see {cite}`Salimans2016`):

* Wasserstein GAN (relying on the Wasserstein distance instead of the log-likelihood) {cite}`Arjovsky2017`.
* f-GAN (relying on any f-divergence) {cite}`Nowozin2016`.

But the generator often **collapses**, i.e. outputs non-sense, or always the same image. Hyperparameter tuning is very difficult.

StyleGAN2 from NVIDIA {cite}`Karras2020` is one of the most realistic GAN variant. Check its generated faces at <https://thispersondoesnotexist.com/>.



## Conditional GANs

<div class='embed-container'><iframe src='https://www.youtube.com/embed/WxVKJfwPjw4' frameborder='0' allowfullscreen></iframe></div>

### cGAN

The generator can also get additional **deterministic** information to the latent space, not only the random vector $z$. One can for example provide the **label** (class) in the context of supervised learning, allowing to generate many **new** examples of each class: data augmentation using a conditional GAN {cite}`Mirza2014`. One could also provide the output of a pre-trained CNN (ResNet) to condition on images.

```{figure} ../img/cgan.png
---
width: 60%
---
cGAN {cite}`Mirza2014`.
```


```{figure} ../img/dcgan_network.jpg
---
width: 100%
---
cGAN conditioned on text {cite}`Reed2016`.
```

```{figure} ../img/dcgan-textimage.jpg
---
width: 100%
---
cGAN conditioned on text {cite}`Reed2016`.
```


### pix2pix

cGAN can be extended to have an autoencoder-like architecture, allowing to generate images from images. **pix2pix** {cite}`Isola2018` is trained on pairs of similar images in different domains. The conversion from one domain to another is easy in one direction, but we want to learn the opposite.

```{figure} ../img/dcgan-imageimage.jpg
---
width: 100%
---
pix2pix {cite}`Isola2018`.
```

The goal of the generator is to convert for example a black-and-white image into a colorized one. It is a deep convolutional autoencoder, with convolutions with strides and transposed convolutions (SegNet-like).

```{figure} ../img/pix2pix-generator1.png
---
width: 100%
---
pix2pix generator. Source: <https://affinelayer.com/pix2pix/>.
```

```{figure} ../img/pix2pix-generator2.png
---
width: 100%
---
Blocks of the pix2pix generator. Source: <https://affinelayer.com/pix2pix/>.
```

In practice, it has a **U-Net** architecture with skip connections to generate fine details.

```{figure} ../img/pix2pix-generator3.png
---
width: 100%
---
pix2pix generator with skip connections. Source: <https://affinelayer.com/pix2pix/>.
```


The discriminator takes a **pair** of images as input: input/target or input/generated. It does not output a single value real/fake, but a 30x30 "image" telling how real or fake is the corresponding **patch** of the unknown image. Patches correspond to overlapping 70x70 regions of the 256x256 input image. This type of discriminator is called a **PatchGAN**.

```{figure} ../img/pix2pix-discriminator-principle.png
---
width: 50%
---
pix2pix discriminator. Source: <https://affinelayer.com/pix2pix/>.
```

```{figure} ../img/pix2pix-discriminator.png
---
width: 100%
---
pix2pix discriminator. Source: <https://affinelayer.com/pix2pix/>.
```

The discriminator is trained like in a regular GAN by alternating input/target or input/generated pairs.


```{figure} ../img/pix2pix-discriminator-training.png
---
width: 100%
---
pix2pix discriminator training. Source: <https://affinelayer.com/pix2pix/>.
```

The generator is trained by maximizing the GAN loss (using gradients backpropagated through the discriminator) but also by minimizing the L1 distance between the generated image and the target (supervised learning).

$$
    \min_G \max_D V(D, G) = V_\text{GAN}(D, G) + \lambda \, \mathbb{E}_\mathcal{D} [|T - G|]
$$

```{figure} ../img/pix2pix-generator-training.png
---
width: 100%
---
pix2pix generator training. Source: <https://affinelayer.com/pix2pix/>.
```

### CycleGAN : Neural Style Transfer

The drawback of pix2pix is that you need **paired** examples of each domain, which is sometimes difficult to obtain. In **style transfer**, we are interested in converting images using unpaired datasets, for example realistic photographies and paintings. **CycleGAN** {cite}`Zhu2020` is a GAN architecture for neural style transfer.

```{figure} ../img/img_translation.jpeg
---
width: 70%
---
Neural style transfer requires unpaired domains {cite}`Zhu2020`.
```


```{figure} ../img/doge_starrynight.jpg
---
width: 100%
---
Neural style transfer. Source: <https://hardikbansal.github.io/CycleGANBlog/>
```


Let's suppose that we want to transform **domain A** (horses) into **domain B** (zebras) or the other way around. The problem is that the two datasets are not paired, so we cannot provide targets to pix2pix (supervised learning). If we just select any zebra target for a horse input, pix2pix would learn to generate zebras that do not correspond to the input horse (the shape may be lost).
How about we train a second GAN to generate the target?


```{figure} ../img/cycle-gan-zebra-horse-images.jpg
---
width: 50%
---
Neural style transfer between horses and zebras. Source: <https://towardsdatascience.com/gender-swap-and-cyclegan-in-tensorflow-2-0-359fe74ab7ff>
```

**Cycle A2B2A**

* The A2B generator generates a sample of B from an image of A.
* The B discriminator allows to train A2B using real images of B.
* The B2A generator generates a sample of A from the output of A2B, which can be used to minimize the L1-reconstruction loss (shape-preserving). 

```{figure} ../img/cyclegan-AB.jpeg
---
width: 100%
---
Cycle A2B2A. Source: <https://towardsdatascience.com/gender-swap-and-cyclegan-in-tensorflow-2-0-359fe74ab7ff>
```

**Cycle B2A2B**

In the B2A2B cycle, the domains are reversed, what allows to train the A discriminator.

```{figure} ../img/cyclegan-BA.jpeg
---
width: 100%
---
Cycle B2A2B. Source: <https://towardsdatascience.com/gender-swap-and-cyclegan-in-tensorflow-2-0-359fe74ab7ff>
```

This cycle is repeated throughout training, allowing to train both GANS concurrently.

```{figure} ../img/cycleGAN2.jpg
---
width: 100%
---
CycleGAN. Source: <https://github.com/junyanz/CycleGAN>.
```

```{figure} ../img/cycleGAN3.jpg
---
width: 100%
---
CycleGAN. Source: <https://github.com/junyanz/CycleGAN>.
```

```{figure} ../img/cycleGAN4.jpg
---
width: 100%
---
CycleGAN. Source: <https://github.com/junyanz/CycleGAN>.
```


<div class='embed-container'><iframe src='https://www.youtube.com/embed/fu2fzx4w3mI' frameborder='0' allowfullscreen></iframe></div>
