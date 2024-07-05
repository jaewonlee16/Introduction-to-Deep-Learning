# Exercise 1 - Training generative model to generate MNIST digits

## Overview
This exercise focuses on implementing and training AutoEncoder (AE) and Variational AutoEncoder (VAE) models, along with their variant, beta-VAE. The primary goal is to minimize the reconstruction error and the Kullback-Leibler (KL) divergence to learn efficient latent representations of the input data.

## Structure

### Encoder Class
The `Encoder` class is responsible for encoding input image tensors into latent vectors. It includes:
- Four convolutional layers.
- Two fully-connected layers (`self.fc_mean` and `self.fc_logvar`).
- A `reparameterize` function used exclusively for VAE to sample latent vectors from a Gaussian distribution.

### Decoder Class
The `Decoder` class is responsible for reconstructing the input images from the latent vectors. It consists of:
- An `input_layer` to reshape the latent vector.
- A series of convolutional layers (inverse of the `Encoder`).
- A `last_layer` to produce the final output.

### Loss Functions
- **AutoEncoder**: Uses only the reconstruction loss.
- **Variational AutoEncoder**: Uses VAE loss, which includes both reconstruction loss and KL divergence.

### Training Functions
- `training_VAE`: Function to train both AE and VAE models. It involves passing the input tensor through the encoder and decoder, computing the loss, and backpropagating to update parameters.

## Implementation Details

### Encoder Class

```python
class Encoder(nn.Module):
    def __init__(self, hidden_dims, latent_dim, model_name='AE'):
        super(Encoder, self).__init__()
        self.model_name = model_name
        # Define model layers
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(1, hidden_dims[0], 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(hidden_dims[1], hidden_dims[2], 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(hidden_dims[2], hidden_dims[3], 4, 2, 1),
            nn.LeakyReLU(0.01),
        )
        # Fully connected layers
        self.fc_mean = nn.Linear(hidden_dims[3] * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[3] * 4 * 4, latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mean(x)
        if self.model_name == 'VAE':
            logvar = self.fc_logvar(x)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z
        return mu
```

### Decoder Class

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, expand_dim):
        super(Decoder, self).__init__()
        self.input_layer = nn.Linear(latent_dim, hidden_dims[-1] * expand_dim * expand_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-2], 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(hidden_dims[-2], hidden_dims[-3], 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(hidden_dims[-3], hidden_dims[-4], 4, 2, 1),
            nn.LeakyReLU(0.01),
        )
        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-4], 1, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = x.view(x.size(0), -1, self.expand_dim, self.expand_dim)
        x = self.decoder(x)
        return self.last_layer(x)
```

### Training Function

```python
def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            img, _ = data
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

## Results
After completing the implementation, the models are trained on the MNIST dataset. The following outputs are generated:
- Reconstruction results for AE and VAE.
- Loss curves for each model.
- Visualizations of the latent space.

## Conclusion
This exercise provides a comprehensive understanding of the implementation and training of AutoEncoders and Variational AutoEncoders. The results demonstrate the effectiveness of these models in learning compact representations of data, with VAE offering additional advantages through probabilistic interpretation.

For any further queries or issues, please contact the author.

---

