from builtins import range
import numpy as np

 

class Conv(object):
  @staticmethod
  def _find_roi(x, i, j, FH, FW, stride):
    """
    Extract the receptive field  or region of interest (ROI) from the single input tensor `x` for a given position (i, j) in the output feature map.
    The receptive field is the segment of the input tensor that the convolutional filter will overlap with during the convolution process. 
    Given a specific position (i, j) in the output feature map, the function determines and extracts the correspoding receptive field from the input tensor.
    Input:
        x (numpy.ndarray): Input data of shape (H, W, C).
        i (int): Vertical index of the output feature map.
        j (int): Horizontal index of the output feature map.
        FH (int): Height of the convolutional filter.
        FW (int): Width of the convolutional filter.
        stride (int): The number of pixels between adjacent receptive fields in the
                      horizontal and vertical directions.
    Returns:
       out: The receptive field with shape (FH, FW, C) from the input tensor `x` for a given position (i, j) in the output feature map.
    """
    out=None
    #########################WRITE YOUR CODE###################################
    # TODO: Implement the extracting roi function for the given input                
    ###########################################################################
    
    pass
    
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return out  
    
  @staticmethod
  def naive_forward(x, w, b, conv_param):      
    """An implementation of the forward pass for a convolutional layer.

    The input consists of N data points, height H and
    width W, each with C channels. We convolve each input with F different filters, where each filter
    has height HH and width WW and spans all C channels.
    
    Input:
    - x: Input data of shape (N, H, W, C)
    - w: Filter weights of shape (F, FH, FW, C)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, H', W', F) where H' and W' are given by
      H' = 1 + (H + 2 * pad - FH) // stride
      W' = 1 + (W + 2 * pad - FW) // stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #########################WRITE YOUR CODE###################################
    # TODO: Implement the convolutional forward pass.                         #
    # Here, you do not have to consider computational efficiency              #
    # You should use `Conv._find_roi` function when implementing              #
    # `Conv.naive_forward` function.                                          #
    ###########################################################################
    
    
    pass


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


  @staticmethod
  def forward(x, w, b, conv_param):      
    """An efficient implementation of the forward pass for a convolutional layer.

    The input consists of N data points, height H and
    width W, each with C channels. We convolve each input with F different filters, where each filter
    has height HH and width WW and spans all C channels.

    Input:
    - x: Input data of shape (N, H, W, C)
    - w: Filter weights of shape (F, FH, FW, C)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
         - 'stride': The number of pixels between adjacent receptive fields in the
           horizontal and vertical directions.
         - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, H', W', F) where H' and W' are given by
      H' = 1 + (H + 2 * pad - FH) // stride
      W' = 1 + (W + 2 * pad - FW) // stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #########################WRITE YOUR CODE###################################
    # TODO: Implement the efficient convolutional forward pass.               #
    # As we mentioned in .ipynb, you can only consider removing the for-loop  #
    # related to F for the computational efficiency.                          #
    # You should use `Conv._find_roi` function when implementing              #
    # `Conv.forward` function.                                                #
    ###########################################################################
    
    
    pass
    
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


  @staticmethod
  def backward(dout, cache):
    """An implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in Conv.forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #########################WRITE YOUR CODE###################################
    # TODO: Implement the convolutional backward pass.                        # 
    # You do not have to consider computational efficiency here, but          #
    # if you consider it, you will have a chance to earn the bonus point      #
    ###########################################################################
    
    
    pass
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
 
class Pooling(object):
  @staticmethod
  def forward(x, pool_param):
    """An implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, H, W, C)
    - pool_param: dictionary with the following keys:
        - 'pool_size': The size of each pooling region.  
           Here, we only care about square sized pooling (pool_height == pool_weight).
        - 'stride': The distance between adjacent pooling regions
        - 'pool_type': "max" or "avg"

    No padding is necessary here
    Returns a tuple of:
    - out: Output data, of shape (N,  H', W', C) where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    #########################WRITE YOUR CODE###################################
    # TODO: Implement the max-pooling forward pass                            #
    # Here, you do not have to consider computational efficiency              #
    # but, if you consider it, you will have a chance to earn the bonus point #
    ###########################################################################
    
    pass
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """An implementation of the backward pass for a pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #########################WRITE YOUR CODE###################################
    # TODO: Implement the max-pooling backward pass                           #
    # Here, you do not have to consider computational efficiency              #
    # but, if you consider it, you will have a chance to earn the bonus point #
    ###########################################################################
    
    pass

                    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

class BatchNorm(object):
    @staticmethod
    def _compute_means_and_vars(x, axis):
        """
        Computes the mean and variance of the input data.
        
        Inputs:
        - x: Input data.
        - axis: Axis or axes along which the means and variances are computed.

        Returns:
        - means: Computed means along specified axis.
        - vars: Computed variances along specified axis.
        """
        means, vars=None,None
        
        #################################WRITE YOUR CODE#############################################
        # TODO: Implement the function for computing the mean and variance of the input data.      
        #############################################################################################
        
        pass

        
        #############################################################################################
        #                             END OF YOUR CODE                            
        #############################################################################################
        return means, vars

    @staticmethod
    def _normalize_data(x, means, vars,eps):
        """
        Normalizes the input data using the provided means and variances.
        
        Inputs:
        - x: Input data.
        - means: Means used for normalization.
        - vars: Variances used for normalization.
        - eps: Small value to add to variance for numerical stability.

        Returns:
        - out: Normalized input 'x'
        """
        out = None
        #################################WRITE YOUR CODE#############################################
        # TODO: Implement the function for normalizing the data      
        #############################################################################################
        
        pass
        
        #############################################################################################
        #                             END OF YOUR CODE                            
        #############################################################################################
        
        return out

    @staticmethod
    def _scale_and_shift(x, gamma, beta):
        """
        Scales and shifts the normalized data using gamma and beta parameters.
        Inputs:
        - x: Data that has been normalized.
        - gamma: Scaling parameter.
        - beta: Shifting parameter.

        Returns:
        - out: Scaled and shifted data.
        """
        out= None
        #################################WRITE YOUR CODE#############################################
        # TODO: Implement the function for scaling and shifting the data 
        #############################################################################################

        pass

        #############################################################################################
        #                             END OF YOUR CODE                            
        #############################################################################################
        return out

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass of batch normalization for CNN.
        Implement it using above functions (_compute_means_and_vars, _normalize_data, _scale_and_shift)

        Inputs:
        - x: Input data of shape (N, H, W, C)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
            - mode: 'train' or 'test'; required
            - eps: Constant for numeric stability (we set for 1e-5 here)
            - momentum: Constant for running mean / variance. momentum=0 means that
                old information is discarded completely at every time step, while
                momentum=1 means that new information is never incorporated. The
                default of momentum=0.9 should work well in most situations.
            - running_mean: Array of shape (C,) giving running mean of features
            - running_var Array of shape (C,) giving running variance of features

        Returns a tuple of:
        - out: Output data, of shape (N, H, W, C)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None
        
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)
        N, H, W, C = x.shape
        running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
        running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))


        if mode == 'train':
            #################################WRITE YOUR CODE###########################################
            # TODO: Implement the training-time forward pass for spatial batch norm.      
            # - During training, compute the mean and variance from the input minibatch.
            # - Normalize the data using these statistics.
            # - Scale and shift the normalized data with gamma and beta.
            # - Calculate the running mean/ std and store them         
            #
            # - Store the output in the variable out, and intermediates              
            #   that you need for the backward pass should be stored in the cache variable.                                                           
            # For further information, refer to the original paper (https://arxiv.org/abs/1502.03167)  
            #  You should use above functions (_compute_means_and_vars, _normalize_data, _scale_and_shift)
            ##########################################################################################
            
            pass
            
            ###########################################################################################
            #                                  END OF YOUR CODE                            
            ###########################################################################################
        elif mode == 'test':
            #################################WRITE YOUR CODE###########################################
            # TODO: Implement the test-time forward pass for batch norm. 
            # - Use the running mean and variance to normalize the incoming data,   
            #   then scale and shift the normalized data using gamma and beta.      
            # - Store the result in the out variable.                               
            ###########################################################################################
            
            pass
        
            ###########################################################################################
            #                                  END OF YOUR CODE                           
            ###########################################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var        
        
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass of batch normalization for CNN.

        Inputs:
        - dout: Upstream derivatives, of shape (N, H, W, C)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, H, W, C)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None
        #################################WRITE YOUR CODE###########################
        # TODO: Implement the backward pass for spatial batch normalization.      #
        #                                                                         #
        ###########################################################################
        
        pass
    
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return dx, dgamma, dbeta



