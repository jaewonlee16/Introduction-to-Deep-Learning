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
    
    position_x = stride * i
    position_y = stride * j
    out = x[position_x : position_x + FH, position_y : position_y + FW, :]
    
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

    pad = conv_param['pad']
    stride = conv_param['stride']

    padded_x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode = 'constant')
    
    N, H, W, C = x.shape
    F , FH, FW, C= w.shape

    """
    # reshape the filter similar to output
    new_filter = np.zeros((FH, FW, C, F))
    for f_index, f in enumerate(w):
        new_filter[:, :, :, f_index] = f

    # reshape input x similar to output
    new_x = np.expand_dims(padded_x, axis = -1)
    new_x = new_x.repeat(F, axis = -1)
    """

    # out shape
    out_shape = (N, 1 + (H + 2 * pad - FH) // stride,1 + (W + 2 * pad - FW) // stride, F)  

    out = np.zeros(out_shape)



    """
    for n in range(N):
        conv = 0
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                x_roi = Conv._find_roi(new_x[n, :], i, j, FH, FW, stride)
                conv = x_roi * new_filter

                out[n, i, j, :] = conv.sum(axis = 0).sum(axis=0).sum(axis=0) + b

    """

    for n in range(N):
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                for f in range(F):
                    x_roi = Conv._find_roi(padded_x[n, :, :, :], i, j, FH, FW, stride)
                    conv = x_roi * w[f, :, :, :]
                    c_sum = conv.sum(axis = 0).sum(axis = 0)
                    out[n, i, j, f] = (c_sum).sum(axis=0) + b[f]

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
    pad = conv_param['pad']
    stride = conv_param['stride']

    padded_x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode = 'constant')
    
    N, H, W, C = x.shape
    F , FH, FW, C= w.shape

    im2col_filter = w.reshape(F, -1)    
    
    im2col_roi = np.zeros((F, FH * FW * C))
    
    # out shape
    out_shape = (N, 1 + (H + 2 * pad - FH) // stride,1 + (W + 2 * pad - FW) // stride, F)  

    out = np.zeros(out_shape)
    
    for n in range(N):
        for i in range(out_shape[1]):
            for j in range(out_shape[2]):
                x_roi = Conv._find_roi(padded_x[n, :, :, :], i, j, FH, FW, stride)
                im2col_roi[:, :] = x_roi.reshape(-1)
                conv = (im2col_filter * im2col_roi).sum(axis=-1)
                out[n, i, j, :] = conv + b


    
    
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
    
    x = cache[0]
    w = cache[1]
    b = cache[2]
    pad = cache[3]['pad']
    stride = cache[3]['stride']

    padded_x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode = 'constant')
    
    N, H, W, C = x.shape
    F , FH, FW, C= w.shape

    OH = dout.shape[1]
    OW = dout.shape[2]
    
    dw = 0
    dx_padded = np.zeros_like(padded_x)
    db = np.zeros(b.shape)

    w_im2col = w.reshape(F, -1).T

    w_im2col_expand = np.zeros((N, FH * FW * C, F))
    w_im2col_expand[:, :, :] = w_im2col[np.newaxis, :, :]

    x_flat = np.zeros((F, N, FH * FW * C))
    """

    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                
                # dw
                dout_im2col = dout[n, i, j, :]
                x_flat[:, :] = padded_x[n, i : i + FH * stride: stride, j : j + FW * stride: stride, :].reshape(-1)
                dw += ((x_flat.T * dout_im2col).T).reshape(w.shape)
                
                #db
                db += dout_im2col
                #dx
                dx_padded[n, i * stride : i * stride + FH, j * stride : j *stride + FW, :] += (w_im2col * dout_im2col).sum(axis=-1).reshape(w.shape[1:])
    """

    for i in range(OH):
        for j in range(OW):
            # dw
            dout_im2col = dout[:, i, j, :]
            x_flat[:, :, :] = padded_x[:, i : i + FH * stride: stride, j : j + FW * stride: stride, :].reshape(N, -1)
            dw += ((x_flat.T * dout_im2col).T).sum(axis=1).reshape(w.shape)

            #db
            db += dout_im2col.sum(axis=0)
    
            #dx
            dx_padded[:, i * stride : i * stride + FH, j * stride : j *stride + FW, :] \
                += (w_im2col * dout_im2col[:, np.newaxis, :]).sum(axis=-1).reshape((N, *w.shape[1:]))

    dx = dx_padded[:, pad : H + pad, pad : W + pad, :]
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
    
    pool_size = pool_param['pool_size']
    stride = pool_param['stride']

    N, H, W, C = x.shape
    H_prime = 1 + (H - pool_size) // stride
    W_prime = 1 + (W - pool_size) // stride

    out = np.zeros((N, H_prime, W_prime, C), dtype=np.float64)

    if pool_param['pool_type'] == 'max':
        for i in range(H_prime):
            for j in range(W_prime):
                out[:, i, j, :] = np.max(x[:, i * stride : i * stride + pool_size, j * stride : j * stride + pool_size, :], axis = (1, 2))
    else:
        for i in range(H_prime):
            for j in range(W_prime):
                out[:, i, j, :] = np.mean(x[:, i * stride : i * stride + pool_size, j * stride : j * stride + pool_size, :], axis = (1, 2))


    
    
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
    
    x , pool_param = cache
    pool_size = pool_param['pool_size']
    stride = pool_param['stride']

    N, H, W, C = x.shape
    H_prime = 1 + (H - pool_size) // stride
    W_prime = 1 + (W - pool_size) // stride

    dx = np.zeros_like(x)
    denominator = pool_size * pool_size
    max_x = np.zeros((N, 1, 1, C))
    dout_expand = np.zeros((N, 1, 1, C))

    if pool_param['pool_type'] == 'max':
        for i in range(H_prime):
            for j in range(W_prime):
                pooled_x = x[:, i * stride : i * stride + pool_size, j * stride : j * stride + pool_size, :]
                max_x[:, 0, 0, :] = np.max(pooled_x, axis=(1, 2))
                argmax = (pooled_x == max_x)
                dout_expand[:, 0, 0, :] = dout[:, i, j, :]
                dx[:, i * stride : i * stride + pool_size, j * stride : j * stride + pool_size, :] += dout_expand * argmax

    else:
        for n in range(N):
            for i in range(H_prime):
                for j in range(W_prime):
                    dx[n, i * stride : i * stride + pool_size, j * stride : j * stride + pool_size, :] \
                        += np.full((pool_size, pool_size, C), dout[n, i, j, :]) / denominator


                    
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

        means = np.mean(x, axis=axis)
        vars = np.var(x, axis=axis)
        

        
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

        out = (x-means) / np.sqrt(vars + eps)


        
        
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

        out = x * gamma + beta

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
            
            mean, var = BatchNorm._compute_means_and_vars(x, (0, 1, 2))
            x_normalized = BatchNorm._normalize_data(x, mean, var, eps)
            out = BatchNorm._scale_and_shift(x_normalized, gamma, beta)

            # momentum
            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var

            #cache
            cache = (x, x_normalized, mean, var, gamma, beta, eps)
            
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
            
            x_normalized = BatchNorm._normalize_data(x, running_mean, running_var, eps)
            out = BatchNorm._scale_and_shift(x_normalized, gamma, beta)
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
        x, x_normalized, mean, var, gamma, beta, eps = cache
        N , H, W, C = x.shape
        m = N * H * W

        dgamma = (dout * x_normalized).sum(axis=(0, 1, 2))
        dbeta = dout.sum(axis=(0, 1, 2))

        dx_normalized = dout * gamma

        dx_mu1 = dx_normalized / np.sqrt(var + eps)

        

        dvar = np.sum(dx_normalized * (x - mean) * (-0.5) * np.power(var + eps, -1.5), axis=(0, 1, 2))
        

        dmean = np.sum(dx_normalized * (-1.0) / np.sqrt(var + eps), axis=(0, 1, 2)) + dvar * np.mean(-2.0 * (x - mean), axis=(0, 1, 2))
        dx = dx_normalized / np.sqrt(var + eps) + dvar * 2.0 * (x - mean) / m + dmean / m


        
    
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return dx, dgamma, dbeta
