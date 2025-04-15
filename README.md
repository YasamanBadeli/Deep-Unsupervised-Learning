# Deep-Unsupervised-Learning

# 👗 Fashion Style Generation using Conditional Variational Autoencoder (CVAE)

# 📚 Dataset
This project uses a Fashion Dataset consisting of clothing images and their associated style labels.

Images are stored in the images/ folder.

A styles.csv file provides metadata, including:

filename: the name of each image file

style: the fashion category or type

The images are converted to grayscale and resized to 28×28 pixels for simplicity and faster training.

# 🎯 Project Goal
This project aims to develop a deep learning pipeline for classifying and recoloring fashion editorial products using Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs).

The workflow consists of two main stages:

## Unsupervised Classification:
Fashion items are analyzed using deep feature extraction and clustering to classify them based on style, material, and color attributes. This creates a structured understanding of the dataset by identifying dominant colors and visual patterns.

## Recoloring via Generation:
After classification, the system generates new color variants of the same fashion items while preserving their structure and texture.

VAEs learn latent representations for smooth transformations across colors.

GANs (e.g., Pix2Pix or CycleGAN) enhance the realism and quality of recolored outputs.

This approach enables automated fashion augmentation, supporting digital styling and editorial design.

# 🛠️ Key Components
## ✅ Dataset
. Images: Fashion item photos

. Labels: Style categories (e.g., dress, shirt, jeans)

## ✅ Model
. Encoder: Learns latent representations conditioned on label

. Decoder: Generates images from latent space and label

## ✅ Loss Function
. Reconstruction Loss: Binary cross-entropy between input and output

. KL Divergence: Encourages the latent space to follow a normal distribution

## ✅ Visualization
. Generated samples are visualized:

  . Each sample is conditioned on a specific style

  . Output is a grid of generated fashion items

# 🚀 Getting Started
🔧 Prerequisites
Python 3.8+

PyTorch

torchvision

pandas

matplotlib

Pillow

# 💡 Usage
1. Preprocessing
Reads and resizes images to 28×28 grayscale

Labels are one-hot encoded

2. Training
bash
Copy
Edit
python cvae_fashion.py
Trains the CVAE model using image-label pairs

Saves the trained model to models/cvae.pth

3. Visualization
After training, the script generates samples for each style label

Displays results in a side-by-side image grid


# 
Let me know if you’d like:

📁 A zipped folder with all files structured

🖼️ Help designing a banner or adding sample images

🌐 A version ready to publish on GitHub

Happy coding! 🌟





