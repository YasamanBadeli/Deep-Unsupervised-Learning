from gettext import install


# Install the required package using pip in the terminal
# Install the required package using pip in the terminal
# pip install openai
# pip install openai

import streamlit as st
import pandas as pd
import requests
import os
import json
from PIL import Image
from io import BytesIO
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128*16*16, latent_dim)
        self.fc_logvar = nn.Linear(128*16*16, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 128*16*16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x): 
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar): 
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z): 
        return self.decoder(self.decoder_input(z))

    def forward(self, x): 
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img
    
st.title("🖼️ Fashion Image VAE App")

csv_file = st.file_uploader("Upload CSV with 'filename' and 'link' columns", type=['csv'])
if csv_file:
    df = pd.read_csv(csv_file)
    sample_df = df.sample(n=1000).reset_index(drop=True)
    st.write(sample_df.head())
    
    # Convert to JSON format
    image_list = [{"imageId": row["filename"], "url": row["link"]} for _, row in sample_df.iterrows()]
    with open("sampled_images.json", "w") as f: json.dump({"images": image_list}, f)
    st.success("Sampled and saved 1000 images.")

# Step 2: Download and Resize
if st.button("Download & Resize Images"):
    output_dir = "downloaded_images"
    os.makedirs(output_dir, exist_ok=True)
    with open("sampled_images.json") as f:
        entries = json.load(f)["images"]

    success = 0
    for item in tqdm(entries):
        try:
            response = requests.get(item["url"], timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((128, 128))
            img.save(os.path.join(output_dir, f"{item['imageId']}.jpg"))
            success += 1
        except:
            continue
    st.success(f"Downloaded and resized {success} images!")

# Step 3: Train VAE
if st.button("Train VAE on Downloaded Images"):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolderDataset("downloaded_images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    st.write("Training...")

    for epoch in range(3):  # Simple 3 epochs
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss = ((batch - recon)**2).sum() + 0.5 * torch.sum(mu**2 + torch.exp(logvar) - 1 - logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        st.write(f"Epoch {epoch+1}: Loss = {total_loss:.2f}")
    torch.save(model.state_dict(), "vae_model.pth")
    st.success("VAE trained and model saved.")

# Step 4: Visualize Reconstructions
if st.button("Show Reconstructions"):
    model = VAE().to(device)
    model.load_state_dict(torch.load("vae_model.pth", map_location=device))
    model.eval()

    dataset = ImageFolderDataset("downloaded_images", transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    batch = next(iter(dataloader)).to(device)
    with torch.no_grad():
        recon, _, _ = model(batch)

    st.subheader("Original vs Reconstructed Images")
    for i in range(8):
        col1, col2 = st.columns(2)
        orig_img = transforms.ToPILImage()(batch[i].cpu())
        recon_img = transforms.ToPILImage()(recon[i].cpu())
        col1.image(orig_img, caption="Original")
        col2.image(recon_img, caption="Reconstructed")
