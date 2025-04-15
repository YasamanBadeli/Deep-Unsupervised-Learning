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
The objective is to build a Conditional Variational Autoencoder (CVAE) that can:

# . Learn latent representations of fashion images

# . Generate new images conditioned on specific clothing styles

# . Explore the structure of fashion latent space

This allows both style-conditioned generation and a deeper understanding of visual characteristics of fashion items.

# 🛠️ Key Components
# ✅ Dataset
. Images: Fashion item photos

. Labels: Style categories (e.g., dress, shirt, jeans)

# ✅ Model
. Encoder: Learns latent representations conditioned on label

. Decoder: Generates images from latent space and label

# ✅ Loss Function
. Reconstruction Loss: Binary cross-entropy between input and output

. KL Divergence: Encourages the latent space to follow a normal distribution

# ✅ Visualization
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





