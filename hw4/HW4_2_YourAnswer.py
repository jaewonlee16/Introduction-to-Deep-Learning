import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
from PIL import Image
from utils import *
from HW4_fid_YourAnswer import calculate_fid,PartialInceptionNetwork

class Generator(nn.Module):
    def __init__(self,latent_dim = 100, nc = 3):
        """"
        Args:
            input_shape : shape of the input image
            latent_dim : size of the latent z vector
            nc : number of channels in the training images. For color images this is 3
        """
        self.latent_dim = latent_dim
        super(Generator, self).__init__()
        self.model = None
        # TODO : Build the generator model
        ##############################################
        ############### YOUR CODE HERE ###############
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, nc * 16 * 16),
            nn.Tanh()
        )
        self.nc = nc
        ############### YOUR CODE HERE ###############
        ##############################################

    def forward(self, input):
        """
        Forward pass of the generator.

        Args:
            input : the input to the generator (latent vector) (type : torch.Tensor, size : (batch_size, latent_dim))
        Returns:
            generated_img : the output of the generator (image) (type : torch.Tensor, size : (batch_size, 1, 16, 16))
        """
        assert input.shape == (input.shape[0], self.latent_dim), f"input shape must be (batch_size, latent_dim), not {input.shape}"
        
        generated_img = None
        # TODO : Implement the forward pass of the generator
        ##############################################
        ############### YOUR CODE HERE ###############
        generated_img = self.model(input)
        generated_img = generated_img.view(-1, self.nc, 16, 16)

        ############### YOUR CODE HERE ###############
        ##############################################
        return generated_img

class Discriminator(nn.Module):
    def __init__(self, nc=3):
        """
        Args:
            256 : size of feature maps in generator
            nc : number of channels in the training images. For color images this is 3
        """
        super(Discriminator, self).__init__()
        self.nc = nc

        self.model = None
        # TODO : Build the discriminator model
        ##############################################
        ############### YOUR CODE HERE ###############
        self.model = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )
        
        ############### YOUR CODE HERE ###############
        ##############################################

    def forward(self, input):
        """
        Forward pass of the discriminator.
        
        Args:
            input : the input to the discriminator (image) (type : torch.Tensor, size : (batch_size, 1, 16, 16))
        Returns:
            output : the output of the discriminator (probability of being real) (type : torch.Tensor, size : (batch_size))
        """
        assert input.shape == (input.shape[0], self.nc, 16, 16), f"input shape must be (batch_size, 1, 16, 16), not {input.shape}"
        output = None
        ##############################################
        # Detail : The output of the discriminator should be (batch_size, 1)
        ############### YOUR CODE HERE ###############
        output = self.model(input)
        
        ############### YOUR CODE HERE ###############
        ##############################################
        return output
    
class cGenerator(nn.Module):
    def __init__(self,input_shape = (1,28,28),
          latent_dim = 100,  num_classes = 10, nc = 3):
        """"
        Args:
            input_shape : shape of the input image
            latent_dim : size of the latent z vector
            num_classes : number of classes
            nc : number of channels in the training images. For color images this is 3
        """
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        super(cGenerator, self).__init__()

        self.model = None
        # TODO : Build the conditional generator model with the given architecture using nn.Sequential
        ##############################################
        ############### YOUR CODE HERE ###############
        
        ############### YOUR CODE HERE ###############
        ##############################################

    def forward(self, input,label):
        """
        Args:
            input : random noise z
            label : label for the image
        Returns:
            generated image
        """
        batch_size = input.shape[0]
        output = None
        ##############################################
        # Detail : 
        # Forward input into your own model
        ############### YOUR CODE HERE ###############
        
        ############### YOUR CODE HERE ###############
        ##############################################
        return output
  
    
class cDiscriminator(nn.Module):
    def __init__(self,input_shape = (3,16,16), nc = 3, num_classes=10):
        """"
            input_shape : shape of the input image
            nc : number of channels in the training images. For color images this is 3
            num_classes : number of classes
        """
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.num_classes = num_classes
        self.nc = nc
        super(cDiscriminator, self).__init__()
        self.label_embedding = None
        self.model = None
        # TODO : Build the conditional discriminator model with the given architecture using nn.Sequential
        ##############################################
        ############### YOUR CODE HERE ###############
        
        ############### YOUR CODE HERE ###############
        ##############################################

    def forward(self, input,label): 
        assert input.shape == (input.shape[0], self.nc, 16, 16), f"input shape must be (batch_size, 1, 16, 16), not {input.shape}"
        output = None
        ##############################################
        # Detail : Forward input into your own model
        ############### YOUR CODE HERE ###############
        
        ############### YOUR CODE HERE ###############
        ##############################################
        return output

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
        #          You need to define your own transform.
        ############### YOUR CODE HERE ###############
        
        ############### YOUR CODE HERE ###############
        ##############################################

        data = np.load("./sprites_1788_16x16.npy")
        label = np.load("./sprite_labels_nc_1788_16x16.npy")

        self.dataset = None
        # TODO : Load the dataset with implemented class
        ##############################################
        # Detail : torch.utils.data.TensorDataset for the dataset sprites_1788_16x16.npy
        #          The arguments for the TensorDataset are torch.Tensor type (change the type with torch.Tensor)
        #          Apply self.transform which defined above
        #          Check the shape of the data
        #          The shape of the data should be (1788, 3, 16, 16)
        ############### YOUR CODE HERE ###############
        
        ############### YOUR CODE HERE ###############
        ##############################################

        self.dataloader = None
        # TODO : Create the dataloader with implemented class
        ##############################################
        # Detail : Use torch.utils.data.DataLoader for the dataloader
        #         Set batch_size as self.batch_size
        #         Set shuffle is applied for the train dataloader, but not for the test dataloader
        ############### YOUR CODE HERE ###############
        
        ############### YOUR CODE HERE ###############
        ##############################################
    def __len__(self):
        return len(self.dataloader)
    def __iter__(self):
        return iter(self.dataloader)
    def __getitem__(self, idx):
        return self.dataloader[idx]
    
def loss_function(prob, label='fake'):
    """
    Return the loss function used for training the GAN.
    Args:
        prob : the probability of the input being real or fake (output of the discriminator) 
                (type : torch.Tensor, size : (batch_size, 1))
        label : either 'fake' or 'real' to indicate whether the input is (intend to be) fake or real 
                (type : str)"
    Returns:
        loss : the loss value (type : torch.Tensor, size : torch.Size([]))
    """
    loss = 0.0
    batch_size = prob.shape[0]
    ##############################################
    ############### YOUR CODE HERE ###############
    
    ############### YOUR CODE HERE ###############
    ##############################################
    assert loss.shape == torch.Size([]), f"loss shape must be torch.Size([]), not {loss.shape}"
    
    return loss

class training_GAN:
    def __init__(self, train_loader, generator, discriminator, device,
                 config, fid_score_on = False, save_model = False, img_show = False,
                 evaluation_on = False):
        """"
        Initialize the training_GAN class.

        Args:
            train_loader : the dataloader for training dataset (type : torch.utils.data.DataLoader)
            generator : the generator model (type : nn.Module)
            discriminator : the discriminator model (type : nn.Module)
            device : the device where the model will be trained (type : torch.device)
            config : the configuration for training (type : SimpleNamespace)
            fid_score_on : whether to calculate the FID score or not (type : bool)
            save_model : whether to save the model or not (type : bool)
            img_show : whether to show the generated image or not for each epoch (type : bool)
            evaluation_on : whether to evaluate the model or not (type : bool). It turns on for the last epoch.
        """
        self.train_loader = train_loader
        self.num_epochs = config.epoch
        self.lr = config.lr
        self.latent_dim = config.latent_dim
        self.batch_size = config.batch_size
        self.device = device
        self.fid_score_on = fid_score_on
        self.img_show = img_show
        self.evaluation_on = evaluation_on
        self.config = config

        self.generated_img = []
        self.G_loss_history = []
        self.D_loss_history = []
        self.system_info = getSystemInfo()
        self.save_model = save_model
        self.model_name = 'GAN'
        self.generator = generator
        self.discriminator = discriminator
        
        self.optimizer_G = None
        self.optimizer_D = None
        # TODO : Initialize the optimizer for generator and discriminator
        ##############################################
        # Detail : Find the proper optimizer in torch.optim
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
    def one_iter_train(self, images,label,noise=None):
        """
        Train the GAN model for one iteration.

        Args:
            images : the real images (type : torch.Tensor, size : (batch_size, 1, 16, 16))
            label : the label of the real images (type : torch.Tensor, size : (batch_size))
            noise : the random noise z (type : torch.Tensor, (batch_size, latent_dim)
                - If noise is None, then generate the random noise z inside the function. 
                - In general, noise is None. It is for testing the model with fixed noise,
        Returns:
            loss_G : the loss value for generator (type : float)
            loss_D : the loss value for discriminator (type : float)
        """
        loss_D = 0.0
        loss_G = 0.0
        # TODO : Update the discriminator and generator
        ##############################################
        # Consider when the noise is None, then generate the random noise z inside the function.
        # Update the discriminator
        # Detail : Feed the random noise to the generator and get the fake images with no gradient
        #          Then feed the fake images to the discriminator and get the probability of the fake images 
        #          Also feed the real images to the discriminator and get the probability  of the real images 
        #          Calculate the loss of the discriminator ('loss_D')
        #          Backpropagate the loss and update the discriminator
        # Update the generator
        # Detail : Feed the random noise to the generator and get the fake images
        #          Then feed the fake images to the discriminator and get the probability of the fake images 
        #          Calculate the loss ('loss_G')
        #          Backpropagate the loss and update the generator
        ############### YOUR CODE HERE ###############

        ############### YOUR CODE HERE ###############
        ##############################################


        return {
                "loss_G" : loss_G.item(),
                "loss_D" : loss_D.item()
                }
    def get_fake_images(self, z,labels):
        with torch.no_grad():
            out = self.generator(z)
        return out
    def FID_score(self,network, test_images, fake_images, batch_size):
        from fid_score import FID_score
        fid = FID_score(network,test_images,fake_images, batch_size)
        return fid
    def train(self):
        """
        Train the GAN model.
        """
        if self.fid_score_on:
            inception_network = PartialInceptionNetwork().cuda()
        else:
            inception_network = None
        try : 
            test_noise = torch.load("./test_file/test_noise.pth", map_location=self.device)
            test_batch_size = test_noise.shape[0]
            test_data = torch.load("./test_file/img_per_label_sprite.pth", map_location=self.device)
            
            for epoch in range(1,self.num_epochs+1):
                pbar = tqdm.tqdm(enumerate(self.train_loader,start=1), total=len(self.train_loader))
                for i, (images, labels) in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    if epoch == 1 and i == 1:
                        # save the generated images before training
                        fake_images = self.get_fake_images(test_noise,test_data[5]['label'])
                        grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True).detach().cpu().permute(1,2,0).numpy()
                        self.generated_img.append((grid_img* 255).astype('uint8'))
                    self.generator.train()
                    self.discriminator.train()
                    results = self.one_iter_train(images,labels)
                    self.generator.eval()
                    self.discriminator.eval()
                    loss_G, loss_D = results['loss_G'], results['loss_D']
                    self.G_loss_history.append(loss_G)
                    self.D_loss_history.append(loss_D)
                    

                    pbar.set_description(
                        f"Epoch [{epoch}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Loss_D: {loss_D:.6f}, Loss_G: {loss_G:.6f}")
                
                
                fake_images = self.get_fake_images(test_noise,test_data[5]['label'])
                grid_img = torchvision.utils.make_grid(fake_images[:16], nrow=4, normalize=True).detach().cpu().permute(1,2,0).numpy()
                self.generated_img.append((grid_img* 255).astype('uint8'))
                # calculate FID score and show the generated images
                if self.model_name == 'GAN':
                    if self.img_show is True:
                        plt.axis('off')
                        plt.imshow(grid_img)
                        plt.pause(0.01)
                    if self.fid_score_on :
                        # your own fid score
                        fid = calculate_fid(inception_network, test_data[5]['img'],fake_images, 16)
                elif self.model_name == 'cGAN':
                    if self.fid_score_on :
                        fid = 0.0
                    for label_idx in range(self.num_classes):
                        test_label = torch.zeros(test_batch_size,self.num_classes, device=self.device) 
                        test_label[:,label_idx] = 1
                        fake_images = self.get_fake_images(test_noise,test_label)
                        if self.img_show is True:
                            plt.subplot(1,self.num_classes,label_idx+1)
                            plt.axis('off')
                            plt.title(f"label : {label_idx}")
                            plt.imshow(torchvision.utils.make_grid(fake_images[label_idx][:16], nrow=4, normalize=True).detach().cpu().permute(1,2,0).numpy())
                        if self.fid_score_on :
                            fid += calculate_fid(inception_network, test_data['im'][test_label],fake_images, 16) / self.num_classes
                    
                    if self.img_show is True:
                        plt.pause(0.01)
                else:
                    raise ValueError("model_name must be either GAN or cGAN, not",self.model_name)
                
                if self.fid_score_on :
                    pbar.write(f"EPOCH {epoch} - FID score : {fid:.6f}")
                
        except KeyboardInterrupt:
            print('Keyboard Interrupted, finishing training...')
        
        # TODO : Save the model
        
        if self.evaluation_on is True:
            if inception_network is None:
                inception_network = PartialInceptionNetwork().cuda()
            if self.model_name == 'GAN':
                fake_images = self.get_fake_images(test_noise,test_data[5]['label'])
                self.fid = self.FID_score(inception_network,test_data[5]['img'],fake_images, 16)
            elif self.model_name == 'cGAN':
                fid = 0.0
                for label_idx in range(self.num_classes):
                    test_label = torch.zeros(test_batch_size,self.num_classes, device=self.device) 
                    test_label[:,label_idx] = 1
                    fake_images = self.get_fake_images(test_noise,test_label)
                    fid += self.FID_score(inception_network,test_data[label_idx],fake_images, 16) / self.num_classes
                self.fid = fid
            else : 
                raise ValueError("model_name must be either GAN or cGAN, not",self.model_name)
        
        if self.evaluation_on is True:
            pbar.write(f"EPOCH {epoch} - FID score : {self.fid:.6f} (evaluation)")
        
        if self.save_model is True:
            self.save_results()
        return {'generator' : self.generator,
                'generator_state_dict' : self.generator.state_dict(),
                'discriminator' : self.discriminator,
                'discriminator_state_dict' : self.discriminator.state_dict(),
                'G_loss_history' : self.G_loss_history,
                'D_loss_history' : self.D_loss_history}
    def save_results(self):
        """
        Save the trained model.
        """
        data = {
            "student_number" : self.config.student_number,
            "generator" : self.generator.state_dict(),
            "discriminator" : self.discriminator.state_dict(),
            "system_info" : self.system_info,
            "epoch" : self.num_epochs,
            "generated_img" : self.generated_img,
        }
        if self.evaluation_on is True:
            data["fid_score"] = self.fid
        if self.save_model is True:
            self.make_gif()
            torch.save(data, f"./{self.model_name}_model.pth")

class training_cGAN(training_GAN):
    def __init__(self, train_loader, generator, discriminator, device, config, fid_score_on=False,save_model=False,img_show=False,evaluation_on=False):
        super().__init__(train_loader, generator, discriminator, device, config, fid_score_on,save_model,img_show,evaluation_on)
        self.num_classes = config.num_classes
        self.model_name = 'cGAN'
    def get_fake_images(self, z, labels):
        return self.generator(z,labels)
    def one_iter_train(self,images,label,noise=None):
        """
        Args:
            images : the real images (type : torch.Tensor, size : (batch_size, 1, 16, 16))
            label : the label of the real images (type : torch.Tensor, size : (batch_size))
            noise : the random noise z (type : torch.Tensor, size : (batch_size, latent_dim))
                - If noise is None, then generate the random noise z inside the function.
                - In general, noise is None. It is for testing the model with fixed noise,
        """
        loss_D = 0.0
        loss_G = 0.0
        # TODO : Update the discriminator and generator
        ##############################################
        # Consider when the noise is None, then generate the random noise z inside the function.
        # Update the discriminator
        # Detail : Feed the random noise and label to the generator and get the fake images with no gradient
        #          Then feed the fake images to the discriminator and get the probability of the fake images 
        #          Also feed the real images to the discriminator and get the probability  of the real images 
        #          Calculate the loss of the discriminator ('loss_D')
        #          Backpropagate the loss and update the discriminator
        # Update the generator
        # Detail : Feed the random noise and label to the generator and get the fake images
        #          Then feed the fake images to the discriminator and get the probability of the fake images 
        #          Calculate the loss ('loss_G')
        #          Backpropagate the loss and update the generator
        ############### YOUR CODE HERE ###############
        
        ############### YOUR CODE HERE ###############
        ##############################################


        return {
                "loss_G" : loss_G.item(),
                "loss_D" : loss_D.item()
                }
    def train(self):
        ret = super().train()
        return ret
