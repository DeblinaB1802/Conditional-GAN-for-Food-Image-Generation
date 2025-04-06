# 🍽️ Conditional GAN for Food Image Generation

This project implements a **Conditional Generative Adversarial Network (cGAN)** using PyTorch to generate high-quality, realistic food images based on class labels from the **Food-101 dataset**.

## 📌 Overview

A Conditional GAN (cGAN) extends the standard GAN by conditioning both the generator and discriminator on auxiliary information such as class labels. This allows the generator to produce images that correspond to a specific class (e.g., sushi, pizza, etc.) rather than generating random images.

In this project, we condition on **food categories** (101 total) using the Food-101 dataset, which contains 101,000 images across 101 food classes.

The core components are:
- A **Conditional Generator** that takes in a noise vector and a class label to generate a synthetic image.
- A **Conditional Discriminator** that tries to distinguish real images from generated ones, while also considering the class label.

## 🎯 Objective

- To build a Conditional GAN architecture using PyTorch.
- To generate class-conditioned synthetic food images from the Food-101 dataset.
- To visualize training progress with generator/discriminator losses.
- To evaluate generation quality by saving and plotting synthetic samples.

## 📁 Dataset

We use the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset provided via `torchvision.datasets.Food101`.

- **Total Images**: 101,000
- **Classes**: 101 food categories
- **Images per Class**: 1,000 (750 train, 250 test)

Images are resized to **256x256** and normalized to the [-1, 1] range.

## 🧠 Model Architecture

### Generator:
- Embeds the class label.
- Concatenates it with a noise vector.
- Passes through a series of transposed convolution layers (ConvTranspose2d) with BatchNorm and ReLU activations.
- Outputs an RGB image (3 channels) via `Tanh()` activation.

### Discriminator:
- Embeds the class label.
- Concatenates it with the input image across channels.
- Uses convolutional layers (Conv2d) with LeakyReLU and BatchNorm to classify real vs fake images.
- Outputs a probability through a `Sigmoid()` activation.

## 🛠️ Training Details

- **Loss Function**: Binary Cross Entropy Loss (BCELoss)
- **Optimizers**: Adam (with β1=0.5, β2=0.999)
- **Noise Dimension**: 256
- **Embedding Dimension**: 100
- **Batch Size**: 64
- **Epochs**: 100
- **Image Size**: 256x256
- **Device**: CUDA / CPU (auto-detected)

During training:
- Generator tries to fool the discriminator into believing the generated image is real.
- Discriminator learns to correctly distinguish real from fake conditioned on class labels.
- Losses are plotted over epochs.
- Final samples are saved to a file and visualized as a grid.

## 📊 Outputs and Visualization

- Generator and Discriminator losses are plotted to assess convergence.
- A grid of 101 generated food images (one per class) is saved as `cgan_samples.png`.
- Generated images are also visualized using matplotlib.

## ✅ Results

- Successfully trained a Conditional GAN to generate realistic food images from random noise.
- Generated images are visually coherent and class-conditional.
- Final generator is able to synthesize 101 different food types.

## 📦 Dependencies

- PyTorch
- TorchVision
- Matplotlib

## 🧪 Future Work

- Improve image sharpness using deeper networks or Progressive Growing GAN.
- Try other architectures like StyleGAN, BigGAN.
- Evaluate image quality using metrics like Inception Score (IS) or FID.
- Deploy model for web-based conditional image generation.

## 📬 Acknowledgments

- Inspired by the original cGAN paper by **Mehdi Mirza and Simon Osindero** (2014). [https://arxiv.org/abs/1411.1784](https://arxiv.org/abs/1411.1784)
- Dataset provided by ETH Zurich’s Computer Vision Lab.

---

Feel free to contribute or fork the repository to build your own version of Conditional GANs!
