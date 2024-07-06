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

Here's a detailed README file for Exercise 2 based on the provided information from your assignment document.

---


# Exercise 2: Training GAN to Generate Pixel Characters and Items

This README file provides a detailed overview of Exercise 2, which involves implementing and training various generative models, including a Generative Adversarial Network (GAN), a Conditional Generative Adversarial Network (cGAN), and evaluating them using the Fréchet Inception Distance (FID) score.

### Objective
The objective of this exercise is to:
1. Implement a Variational AutoEncoder (VAE) and its training.
2. Implement a Generative Adversarial Network (GAN) and its training.
3. Implement a Conditional Generative Adversarial Network (cGAN) and its training.
4. Implement and compute the Fréchet Inception Distance (FID) score for model evaluation.

### Tasks Overview
This exercise consists of several tasks:
1. **Implement GAN and its training.**
2. **Implement Conditional GAN (cGAN) and its training.**
3. **Implement Fréchet Inception Distance (FID) score.**

### Files and Directories
- `HW4_2_YourAnswer.py`: Contains the implementation of GAN, cGAN, and necessary training procedures.
- `HW4_fid_YourAnswer.py`: Contains the implementation for computing the FID score.
- `utils.py`: Contains utility functions to support the main scripts.

### Setting Up the Environment
Depending on your environment (Colab or local setup), select the appropriate section for setting the system path.

#### Setting on Colab
1. **Mount Google Drive:**
    ```python
    from google.colab import drive
    mount_location = '/content/drive'
    drive.mount(mount_location,force_remount=True)
    path = "/content/drive/MyDrive/Colab Notebooks/hw4"
    ```

2. **Set Path and Verify:**
    ```python
    import os, sys
    if os.path.exists(path):
        sys.path.append(path)
        os.chdir(path)
    else:
        raise ValueError("Path does not exist. Set proper path.")
    ```

#### Setting on Local
1. **Set Environment Variables:**
    ```python
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    ```

2. **Set Path and Import Modules:**
    ```python
    path='./'
    import os, sys
    sys.path.append(path)
    ```

### Implementation Details
#### GAN Model
1. **Generator:**
    - Define the model structure in the `__init__` method.
    - Implement the forward pass in the `forward` method.

    Example:
    ```python
    class Generator(nn.Module):
        def __init__(self, latent_dim, nc):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                # layers here
            )

        def forward(self, z):
            return self.model(z)
    ```

2. **Discriminator:**
    - Define the model structure in the `__init__` method.
    - Implement the forward pass in the `forward` method.

    Example:
    ```python
    class Discriminator(nn.Module):
        def __init__(self, nc):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                # layers here
            )

        def forward(self, img):
            return self.model(img)
    ```

#### Training Procedure
1. **Define Hyperparameters:**
    ```python
    config = SimpleNamespace(
        seed=2023,
        input_shape=(3, 16, 16),
        num_classes=5,
        nc=3,
        test_batch_size=64,
        latent_dim=100,
        epoch=20,
        batch_size=32,
        lr=1e-5,
    )
    ```

2. **Set Randomness:**
    ```python
    set_randomness(config.seed)
    ```

3. **Train the Models:**
    - Training loops for both GAN and cGAN.

#### FID Score
1. **Compute FID Score:**
    - Implement the function to calculate FID score in `HW4_fid_YourAnswer.py`.

### Evaluation
- Evaluate the performance of your models based on the generated images' plausibility and quality.
- Use the FID score as the primary metric for evaluation.

### Hyperparameter Tuning
- Experiment with different hyperparameters and model architectures to improve the performance of your GAN and cGAN models.

### Conclusion
This exercise involves implementing complex deep learning models and evaluating their performance using established metrics. Ensure to follow the instructions and complete the TODO sections in the provided Python scripts.
