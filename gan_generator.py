import torch
import torch.nn as nn
import torch.optim as optim

# Prepare synthetic image data (8x8 patterns)
image_size = 8
# Pattern 1: diagonal line from top-left to bottom-right
pattern1 = torch.zeros(image_size, image_size)
for i in range(image_size):
    pattern1[i, i] = 1.0
# Pattern 2: diagonal line from top-right to bottom-left
pattern2 = torch.zeros(image_size, image_size)
for i in range(image_size):
    pattern2[i, image_size - 1 - i] = 1.0

# Create training dataset: 100 images of each pattern
real_images = torch.stack([pattern1 for _ in range(100)] + [pattern2 for _ in range(100)])
real_images = real_images.unsqueeze(1)  # shape [200, 1, 8, 8], add channel dimension

# Define Generator network
class Generator(nn.Module):
    def __init__(self, noise_dim=16, img_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, img_dim * img_dim),
            nn.Sigmoid()  # output values in [0,1] range
        )
        self.img_dim = img_dim
    def forward(self, z):
        out = self.fc(z)
        img = out.view(-1, 1, self.img_dim, self.img_dim)  # reshape to image
        return img

# Define Discriminator network
class Discriminator(nn.Module):
    def __init__(self, img_dim=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim * img_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # outputs probability of being a real image
        )
    def forward(self, img):
        x = img.view(img.size(0), -1)
        return self.model(x)

# Initialize GAN components
noise_dim = 16
G = Generator(noise_dim=noise_dim, img_dim=image_size)
D = Discriminator(img_dim=image_size)
optim_G = optim.Adam(G.parameters(), lr=0.01)
optim_D = optim.Adam(D.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# Training loop
epochs = 100
batch_size = 20
for epoch in range(epochs):
    for i in range(0, len(real_images), batch_size):
        real_batch = real_images[i:i+batch_size]
        if real_batch.size(0) == 0:
            continue

        ## Train Discriminator ##
        real_labels = torch.ones(real_batch.size(0), 1)
        pred_real = D(real_batch)
        loss_real = loss_fn(pred_real, real_labels)

        z = torch.randn(real_batch.size(0), noise_dim)
        fake_images = G(z).detach()
        fake_labels = torch.zeros(real_batch.size(0), 1)
        pred_fake = D(fake_images)
        loss_fake = loss_fn(pred_fake, fake_labels)

        loss_D = loss_real + loss_fake
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        ## Train Generator ##
        z = torch.randn(real_batch.size(0), noise_dim)
        generated_images = G(z)
        pred = D(generated_images)
        loss_G = loss_fn(pred, torch.ones(real_batch.size(0), 1))
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

# Generate a sample image using the trained Generator
G.eval()
sample_noise = torch.randn(1, noise_dim)
generated_image = G(sample_noise).detach().squeeze()
binary_image = (generated_image >= 0.5).int()
print("Generated 8x8 image pattern (1=bright pixel, 0=dark pixel):")
print(binary_image)
