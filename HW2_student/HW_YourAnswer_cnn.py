import torch.nn as nn
from helper_functions import *
from PIL import Image

class numpy_CNN(object):
    """
    A three-layer convolutional network with the following architecture:

    First layer: conv - batch norm- relu - 2x2 max pool
    Second layer:  affine - relu 
    Last layer: affine - softmax

    The network operates on minibatches of data that have shape (N, H, W, C)
    - N: number of images
    - H: image height
    - W: image width
    - C: number of input channels
    
    """

    def __init__(
        self,
        input_dim=(32, 32, 3),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (H, W, C) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ###############################WRITE YOUR CODE############################################
        # TODO:                                                                                  #
        # Initialize weights and biases for the three-layer CNN                                  #
        #   1) Weights should be initialized from a Gaussian centered at 0.0                     #
        #     with standard deviation equal to weight_scale;                                     #
        #   2) biases should be initialized to zero.                                             #
        #   3) gamma should be initialized to one and beta should be initilized to zero          #
        #   3) All weights and biases should be stored in the dictionary self.params.            #
        #   4) Use keys 'W1' and 'b1' for storing weights and biases of conv layer               #   
        #      use keys 'W2' and 'b2' for the weights and biases of the hidden affine layer,     #
        #      use keys 'W3' and 'b3' for the weights and biases of the output affine layer.     #
        #      use keys 'gamma1' and 'beta1' for batch normalization parameter                   #
        #                                                                                        #
        # Here, you can assume that the padding  and stride of the first convolutional layer are #                                         
        # chosen so that **the width and height of the input are preserved**.                    #
        ##########################################################################################
        H, W, C = input_dim

        self.params['W1'] = np.random.normal(0, weight_scale, size=(num_filters, filter_size, filter_size, C))
        self.params['W2'] = np.random.normal(0, weight_scale, size=(H * W * num_filters // 4, hidden_dim))
        self.params['W3'] = np.random.normal(0, weight_scale, size=(hidden_dim, num_classes))
        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)
        self.params['gamma1'] = np.ones(num_filters)
        self.params['beta1'] = np.zeros(num_filters)


        ##########################################################################################
        #                                    END OF YOUR CODE                                    #
        ##########################################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: 
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # Extract Batch Norm parameters
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        bn_param={"mode": "train"}

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = { "pool_size": 2, "stride": 2, "pool_type": "max"}

        scores = None
        if y is None:
            bn_param={"mode": "test"} 
            
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores variable.#
        # Hint: You can use functions in helper_functions.py                       #
        ############################################################################
        conv_out, conv_cache \
                = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, bn_param)
        affine_relu_out, affine_relu_cache = affine_relu_forward(conv_out, W2, b2)
        affine2_out, affine2_cache = affine_forward(affine_relu_out, W3, b3)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # - Compute loss using softmax                                             #
        #   For calculating the loss, you can simply use softmax_loss defined in   #
        #   helper_functions.py                                                    #
        # - You can use other functions in helper_functions.py (affine, relu, ...) # 
        # - Make sure that grads[k] holds the gradients for self.params[k].        #
        # - Don't forget to add L2 regularization!                                 # 
        # - Storing the loss and gradients in the loss and grads variables.        #
        #                                                                          #
        # Hint: You can use functions in helper_functions.py                       #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss(affine2_out, y)
        dout, grads['W3'], grads['b3'] = affine_backward(dout, affine2_cache)
        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, affine_relu_cache)
        _, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_bn_relu_pool_backward(dout, conv_cache)

        # regularization
        weight_squared = np.power(self.params['W1'], 2).sum() \
            + np.power(self.params['W2'], 2).sum() + np.power(self.params['W3'], 2).sum()
        loss = loss + self.reg * 0.5 * weight_squared

        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']
        grads['W3'] += self.reg * self.params['W3']
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads
    
    
    
class pytorch_CNN(nn.Module):
    def __init__(self, num_class=10):
        super(pytorch_CNN, self).__init__()
        """
        A convolutional neural network (using pytorch) with the following architecture:
        - Feature Extraction Layer Block
          - first layer: Conv, Batchnorm, Relu, Max pool
          - second layer:  Conv, Batchnorm, Relu, Max pool
        - Classification Layer Block
          - first layer: Linear, Relu
          - second layer: Linear, Relu
          - final layer: Linear
        """
        # Implement convolutional neural network with pytorch
        #       You have to use torch.nn modules for implementing it
        #       You can use Conv2d, BatchNorm2D, ReLU, MaxPool2d, Linear layer defined in torch.nn module
        
        # TODO: 1) Implement feature extraction layer block
        # =====================================================================================================
        # First Convolutional Layer:
            # Input: Output: 6 channels, Kernel size: 5x5, Padding: 1
            # Batch normalization is applied
            # Activation Function: ReLU (Rectified Linear Unit)
            # MaxPooling: Reduces spatial dimensions (height & width) using a 2x2 kernel with a stride of 2
        # Second Convolutional Layer:
            # Input:  Output: 16 channels, Kernel size: 5x5, Padding: 1
            # Batch normalization is applied
            # Activation Function: ReLU
            # MaxPooling: Further reduces spatial dimensions using a 2x2 kernel with a stride of 2
        #    
        #     2)  Implement classification layer block
        # =====================================================================================================
        # First Linear Layer:  size of each output sample= 4096, followed by ReLU activation
        # Second Linear Layer: size of each output sample= 4096, followed by ReLU activation
        # Output Layer: size of each output sample= #(classes)
        # ========================================== WRITE YOUR CODE ========================================== #
       
        pass
    
        # ===================================================================================================== #

        
    def forward(self, x):
        output=None
        ########################################################################################################
        # TODO: Implement forward pass of pytorch_CNN 
        #########################################################################################################
        
        pass
        
        ##########################################################################################################
        return output

