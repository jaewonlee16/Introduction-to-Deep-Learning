import torch
import torch.nn as nn
import torchvision
import numpy as np
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

class Encoder(nn.Module):
    def __init__(self,in_channels=1, hidden_dims = [16,32,64], latent_dim=2,
              model_name='VAE'):
        """
        Initialize the encoder model.

        Args:
            in_channels : number of channels of input image
            hidden_dims : list of hidden layer dimensions
            latent_dim : dimension of latent vector
            model_name : type of model (beta-VAE or AE)
        """
        super(Encoder, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.model_name = model_name
        self.model = None
        self.fc_mean = None
        self.fc_logvar = None
        # TODO : Fill in the code to define encoder   #
        ##############################################
        # Detail : Use Conv2d layer with kernel size 4, stride 2, padding 1
        #           Use LeakyReLU activation function with default negative slope
        #           -> Repeat the above four layers for each hidden dimension
        #           Use Sequential to define encoder
        #           Use Linear layer to get latent mean `self.fc_mean` and log variance `fc_logvar` for each
        #           dimension of latent vector
        ############### YOUR CODE HERE ###############

        layers = []
        for h_dim in hidden_dims:
            layers.append(
                nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
            )
            in_channels = h_dim
            layers.append(nn.LeakyReLU())

        self.model = nn.Sequential(*layers)

        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        ############### YOUR CODE HERE ###############
        ##############################################

    def reparametrize(self, mu, logvar, eps):
        """
        Returns the reparametrized latent vector.

        Args:
            mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
            logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
            eps : random noise for encoder (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            rp : reparametrized latent vector (type : torch.Tensor, size : (batch_size, latent_dim))
        """
        rp = None
        # TODO : Implement the reparametrization trick
        ##############################################
        # Detail : Fill in the code to reparametrize the latent vector                         
        #          Use reparametrization trick                                   
        ############### YOUR CODE HERE ###############
        rp = mu + eps * torch.exp(0.5 * logvar)
        
        ############### YOUR CODE HERE ###############
        ##############################################
        return rp
    def sample_noise(self, logvar):
        return torch.randn_like(logvar)
    def forward(self, x, eps=None):
        """
        Forward pass of the encoder.

        Args:
            x : the input to the encoder (image) (type : torch.Tensor, size : (batch_size, 1, 16, 16))
            eps : random noise for encoder (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            For VAE, return mu, logvar, rp
                mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
                logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
                rp : reparametrized latent vector (type : torch.Tensor, size : (batch_size, latent_dim))
            For AE, return out
                out : latent vector (type : torch.Tensor, size : (batch_size, latent_dim))
        """
        rp = None
        out = None
        # TODO : Implement the forward pass of encoder
        ##############################################
        # Detail : Fill in the code to forward the input image to encoder and get latent vector 
        #          use reparametrize function to get reparametrized latent vector for VAE
        #          use self.fc_mean as last layer for AE            
        ############### YOUR CODE HERE ###############
        # Forward pass through Convolutional Layers
        x = self.model(x)

        # Flatten the output for Fully Connected Layers
        x = x.view(x.size(0), -1)

        # Compute Mean and Log Variance
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        if self.model_name == 'VAE':
            # Reparametrize for VAE
            if eps is None:
                eps = self.sample_noise(logvar)
            rp = self.reparametrize(mu, logvar, eps)
            return mu, logvar, rp
        elif self.model_name == 'AE':
            # Return the output for AE
            return mu
        
        ############### YOUR CODE HERE ###############
        ##############################################

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims = [16,32,64],expand_dim=4):
        """
        Initialize the decoder model.

        Args:
            latent_dim : dimension of latent vector (type : int)
            hidden_dims : list of hidden layer dimensions (type : list, default : None)
                - reverse order of encoder hidden layer dimensions
            expand_dim : size of the first hidden layer input of self.decoder (type : int)
                - the first hidden layer input of self.decoder is (B, self.hidden_dims[-1], self.expand_dim, self.expand_dim)
        """
        super(Decoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.expand_dim = expand_dim
        self.input_layer = None
        self.decoder = None
        self.last_layer = None
        # TODO : Fill in the code to define decoder   #
        ##############################################
        # Detail :   # self.input_layer 
        #            1 Linear layer
        #            - Use Linear layer to get the first hidden layer
        #                - in features : latent_dim
        #                - out features : self.hidden_dims[-1] * (self.expand_dim ** 2)
        #           # self.decoder
        #           3 ConvTranspose2d layer
        #           - Use ConvTranspose2d layer with kernel size 4, stride 2, padding 1
        #                - in channels : self.hidden_dims[i+1]
        #                - out channels : self.hidden_dims[i]
        #           - Use LeakyReLU activation function with default negative slope
        #           - Repeat the above layers for each hidden dimension (make 3 layers for each hidden dimension)
        #           # self.last_layer 
        #           1 ConvTranspose2d layer
        #           - Use ConvTranspose2d layer with kernel size 3, stride 1, padding 1
        #                - in channels : last hidden dimension
        #                - out channels : 1
        #           - Use Sigmoid activation function
        ############### YOUR CODE HERE ###############
        # Define the input layer
        self.input_layer = nn.Linear(latent_dim, hidden_dims[-1] * (expand_dim ** 2))

        # Define the decoder layers
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i - 1], kernel_size=4, stride=2, padding=1),
            )
            decoder_layers.append(nn.LeakyReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        # Define the last layer
        self.last_layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[0], 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
        )
        
        ############### YOUR CODE HERE ###############
        ##############################################

    def forward(self, x):
        """
        Foward pass of the decoder.

        Args:
            x : the input to the decoder (latent vector) (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            out : the output of the decoder (type : torch.Tensor, size : (batch_size, 1, 16, 16))
        """
        out = None
        # TODO : Implement the forward pass of decoder
        ##############################################
        # Detail : Fill in the code to forward the latent vector to input_layer
        #          and reshape the output with (B, self.hidden_dims[-1], self.expand_dim, self.expand_dim) 
        #          Then, forward the output to self.decoder and self.last_layer      
        ############### YOUR CODE HERE ###############
        out = self.input_layer(x)

        # Reshape the output
        out = out.view(x.size(0), self.hidden_dims[-1], self.expand_dim, self.expand_dim)

        # Forward pass through decoder
        out = self.decoder(out)

        # Forward pass through the last layer
        out = self.last_layer(out)

        return out
        
        ############### YOUR CODE HERE ###############
        ##############################################
        return out

def reconstruction_loss(recon_x, x):
    """
    Returns the reconstruction loss of VAE.
    
    Args:
        recon_x : reconstructed x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
        x : original x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
    Returns:
        recon_loss : reconstruction loss (type : torch.Tensor, size : (1,))
    """
    loss = 0.0
    eps = 1e-18
    batch_size = x.size(0)
    # TODO : Fill in the code to compute the loss 
    ##############################################
    # Don't use torch.nn.functional or other package functions which compute the loss directly            
    ############### YOUR CODE HERE ###############
    
    loss = -torch.sum(x * torch.log(torch.clamp(recon_x, min=1e-10)) + (1 - x) * torch.log(torch.clamp(1 - recon_x, min=1e-10)))
    ############### YOUR CODE HERE ###############
    ##############################################
    loss = loss / batch_size
    return loss

def KLD_loss(mu, logvar):
    """
    Returns the regularization loss of VAE.
    
    Args:
        mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
        logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
    Returns:
        kld_loss : regularization loss (type : torch.Tensor, size : (1,))
    """
    batch_size = mu.size(0)
    kld_loss = None
    # TODO : Fill in the code to compute the loss #
    ##############################################
    # Don't use torch.nn.functional or other package functions which compute the loss directly
    # Detail :  Think about KL Divergence
    ############### YOUR CODE HERE ###############
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ############### YOUR CODE HERE ###############
    ##############################################
    kld_loss = kld_loss / batch_size
    return kld_loss

def loss_function(recon_x, x, mu, logvar,beta=1,return_info=False):
    """
    Returns the loss of beta-VAE.

    Args:
        recon_x : reconstructed x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
        x : original x (type : torch.Tensor, size : (batch_size, 1, 16, 16)
        mu : latent mean (type : torch.Tensor, size : (batch_size, latent_dim))
        logvar : latent log variance (type : torch.Tensor, size : (batch_size, latent_dim))
        beta : beta value for beta-VAE (type : float)
    Returns:
        loss : loss of beta-VAE (type : torch.Tensor, size : (1,))
            - Reconstruction loss + beta * Regularization loss
            Recon_loss : reconstruction loss
               kld_loss : KL divergence loss
    """    
    Recon_loss = reconstruction_loss(recon_x, x)
    kld_loss = KLD_loss(mu, logvar)
    loss = 0.0
    # TODO : Fill in the code to compute the loss #
    ##############################################
    # Detail :  Think about ELBO loss for beta-VAE
    ############### YOUR CODE HERE ###############
    
    Recon_loss = reconstruction_loss(recon_x, x)
    kld_loss = KLD_loss(mu, logvar)
    loss = Recon_loss + beta * kld_loss
    ############### YOUR CODE HERE ###############
    ##############################################
    if return_info:
        return {"loss" : loss,
                "recon_loss" : Recon_loss,
                "kld_loss" : kld_loss}
    else :
        return loss

class dataloader(torch.utils.data.Dataset):
    def __init__(self,train=True, batch_size = 64):
        """"
        Initialize the dataloader class.

        Args:
            train : whether to use training dataset or test dataset (type : bool)
            batch_size : how many samples per batch to load (type : int)
        """
        super(dataloader, self).__init__()
        self.batch_size = batch_size
        self.transform = None
        # TODO : Initialize the transform with implemented class
        ##############################################
        # Detail : Use torchvision.transforms.Compose for the transform
        #         The transform consists of 2 components:
        #               - Resize image to 16x16
        #               - ToTensor
        ############### YOUR CODE HERE ###############
        
        ############### YOUR CODE HERE ###############
        ##############################################

        self.dataset = None
        # TODO : Load the dataset with implemented class
        ##############################################
        # Detail : Use torchvision.datasets.MNIST for the dataset
        #          Apply self.transform which defined above
        #          Set download=True for the dataset
        #            Set root='./data' for the dataset
        ############### YOUR CODE HERE ###############

        ############### YOUR CODE HERE ###############
        ##############################################

        self.dataloader = None
        # TODO : Create the dataloader with implemented class
        ##############################################
        # Detail : Use torch.utils.data.DataLoader for the dataloader
        #         Set batch_size as self.batch_size
        #         Set shuffle is applied for the train dataloader, but not for the test dataloader
        #         Set drop_last=True for the dataloader
        ############### YOUR CODE HERE ###############
        
        ############### YOUR CODE HERE ###############
        ##############################################
    def __len__(self):
        return len(self.dataloader)
    def __iter__(self):
        return iter(self.dataloader)
    def __getitem__(self, idx):
        return self.dataloader[idx]

class training_VAE:
    def __init__(self, train_loader, test_loader, encoder, decoder, device,
                 config, save_img = False, model_name='VAE',beta=1, img_show=False):
        """"
        Initialize the training_VAE class.

        Args:
            train_loader : the dataloader for training dataset (type : torch.utils.data.DataLoader)
            test_loader : the dataloader for test dataset (type : torch.utils.data.DataLoader)
            encoder : the encoder model (type : Encoder)
            decoder : the decoder model (type : Decoder)
            device : the device where the model will be trained (type : torch.device)
            config : the configuration for training (type : SimpleNamespace)
            save_img : whether to save the generated images during training
            model_name : type of model - VAE or AE
                - VAE includes VAE and beta-VAE
            beta : beta value for beta-VAE (type : float)
            img_show : whether to show the generated images during training
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = config.epoch
        self.lr = config.lr
        self.latent_dim = config.latent_dim
        self.batch_size = config.batch_size
        self.device = device

        self.generated_img = []
        self.Recon_loss_history = []
        self.KLD_loss_history = []
        self.system_info = getSystemInfo()
        self.save_img = save_img
        self.model_name = model_name
        if self.model_name == 'beta_VAE':
            self.model_name = f"{self.model_name}_{beta}"
        self.beta = beta
        self.img_show = img_show

        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = None
        # TODO : Initialize the optimizer for encoder and decoder
        ##############################################
        # Detail : Use Adam optimizer with learning rate self.lr 
        #           Include encoder and decoder parameters for optimization
        ############### YOUR CODE HERE ###############

        ############### YOUR CODE HERE ###############
        ##############################################  
    def make_gif(self):
        """
        Save the generated images as a gif file.
        """
        if len(self.generated_img) <= 1:
            print("No frame to save")
            return
        else :
            print("Saving gif file...")
            for i in range(len(self.generated_img)):
                self.generated_img[i] = Image.fromarray(self.generated_img[i])
            self.generated_img[0].save(f"./{self.model_name}_generated_img.gif",
                                save_all=True, append_images=self.generated_img[1:], 
                                optimize=False, duration=700, loop=1) 
    def one_iter_train(self, images,label,eps):
        """
        Train the model for one iteration.

        Args:
            
            images : the input images (type : torch.Tensor, size : (batch_size, 1, 16, 16))
            label : the input labels (type : torch.Tensor, size : (batch_size))
            eps : random noise for encoder (type : torch.Tensor, size : (batch_size, latent_dim))
                - it is used in reparametrization trick
        Returns:
            dict : dictionary of losses
                - VAE :
                recon_loss : the reconstruction loss of the model (type : torch.Tensor, size : (1,))
                kld_loss : the regularization (KL divergence) loss of the model (type : torch.Tensor, size : (1,))
                - AE :
                recon_loss : the reconstruction loss of the model (type : torch.Tensor, size : (1,))
        """
        recon_loss = None
        kld_loss = None
        loss = 0.0
        # TODO : Implement the training code for VAE
        ##############################################
        # Detail : You need to implement the training step for AE, VAE, and beta-VAE
        #          Forward the images to the encoder
        #          Get the latent vector from the encoder
        #          Forward the latent vector to the decoder
        #          Calculate the loss depending on the model type(self.model_name), 'AE','VAE' and 'beta-VAE' 
        #          To get the individual loss (recon_loss, kld_loss),
        #               if the loss is not calculated, set the loss as 0
        #          use loss_function function with return_info = True
        #          Backpropagate the loss and update the encoder and decoder using self.optimizer
        ############### YOUR CODE HERE ###############
        
        
        ############### YOUR CODE HERE ###############
        ##############################################

        return {
                "recon_loss" : recon_loss.item(),
                "kld_loss" : kld_loss.item()
                }
    def get_fake_images(self, image,labels,eps):
        self.encoder.eval()
        self.decoder.eval()
        if self.model_name == 'AE':
            with torch.no_grad():
                rp = self.encoder(image)
                fake_images = self.decoder(rp)
        elif 'VAE' in self.model_name:
            with torch.no_grad():
                mean, logvar, rp = self.encoder(image,eps)
                fake_images = self.decoder(rp)
        else:
            raise NotImplementedError(f"Please choose the model type in ['VAE', 'AE'], not {self.model_name}")
        return fake_images
    def train(self):
        """
        Train the VAE model.
        """
        try : 
            for epoch in range(1,self.num_epochs+1):
                pbar = tqdm(enumerate(self.train_loader,start=1), total=len(self.train_loader))
                for i, (images, labels) in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    eps = torch.randn(self.batch_size, self.latent_dim).to(self.device)
                    if epoch == 1 and i == 1:
                        fake_images = self.get_fake_images(images,labels,eps)
                        grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True).detach().cpu().permute(1,2,0).numpy()
                        self.generated_img.append((grid_img* 255).astype('uint8'))
                    self.encoder.train()
                    self.decoder.train()
                    results = self.one_iter_train(images,labels,eps)
                    self.encoder.eval()
                    self.decoder.eval()
                    recon_loss, kld_loss = results['recon_loss'], results['kld_loss']
                    self.Recon_loss_history.append(recon_loss)
                    self.KLD_loss_history.append(kld_loss)
                    
                    fake_images = self.get_fake_images(images, labels,eps)

                    pbar.set_description(
                        f"Epoch [{epoch}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Total loss : {recon_loss + kld_loss:.6f} Recon Loss: {recon_loss:.6f}, KLD Loss: {kld_loss:.6f}")
                
                grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True).detach().cpu().permute(1,2,0).numpy()
                self.generated_img.append((grid_img* 255).astype('uint8'))
                if self.img_show:
                    plt.imshow(grid_img)
                    plt.pause(0.01)
  
        except KeyboardInterrupt:
            print('Keyboard Interrupted, finishing training...')
        if self.save_img:
            self.make_gif() 
        
        return {'encoder' : self.encoder,
                'encoder_state_dict' : self.encoder.state_dict(),
                'decoder' : self.decoder,
                'decoder_state_dict' : self.decoder.state_dict(),
                'Recon_loss_history' : self.Recon_loss_history,
                'generated_img' : self.generated_img[-1],
                'KLD_loss_history' : self.KLD_loss_history}
