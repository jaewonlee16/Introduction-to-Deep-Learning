# HW2 - CNN implementation using Numpy

## Overview

This assignment involves implementing several layers commonly used in convolutional neural networks (CNNs) using numpy. Specifically, the tasks include:

1. Implementing the forward and backward passes of a convolution layer.
2. Implementing the forward and backward passes of pooling layers (both max and average pooling).
3. Implementing the forward and backward passes for batch normalization in CNNs.
4. Training a three-layer convolutional network on the CIFAR-10 dataset.

This readme provides an overview of the steps taken to complete Exercise 1, which focuses on the convolution layer.

## Files

- `HW_YourAnswer.py`: Contains the implementation of the convolution layer, pooling layers, and batch normalization.
- `HW_YourAnswer_cnn.py`: Used to train the three-layer convolutional network on the CIFAR-10 dataset.
- `utils.py`: Contains utility functions used throughout the assignment.
- Jupyter Notebook: Includes setup and test code to verify the implementations.

## Exercise 1: Convolution Layer with Numpy (No pytorch)
- [2023_SNU_DL_HW2_1.ipynb](https://github.com/jaewonlee16/Introduction-to-Deep-Learning/blob/master/HW2_student/2023_SNU_DL_HW2_1.ipynb)
- [HW_YourAnswer.py](https://github.com/jaewonlee16/Introduction-to-Deep-Learning/blob/master/HW2_student/HW_YourAnswer.py)
- [HW_YourAnswer_cnn.py](https://github.com/jaewonlee16/Introduction-to-Deep-Learning/blob/master/HW2_student/HW_YourAnswer_cnn.py)
  
### Exercise 1.1: Forward Pass of Convolution Layer

#### Implementation Details

The forward pass of the convolution layer involves convolving the input data with a set of filters. The steps are as follows:

1. **Zero-Padding**: We add padding around the input data if required.
2. **Receptive Field Extraction**: For each position in the output, we extract the corresponding receptive field from the input data.
3. **Element-wise Multiplication and Summation**: The receptive field is element-wise multiplied with the filter, and the results are summed up to produce the output.

#### Code Snippets

```python
def naive_forward(self, x, w, b, conv_param):
    N, H, W, C = x.shape
    F, FH, FW, _ = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    
    # Compute the dimensions of the output
    H_out = 1 + (H + 2 * pad - FH) // stride
    W_out = 1 + (W + 2 * pad - FW) // stride
    
    # Initialize the output
    out = np.zeros((N, H_out, W_out, F))
    
    # Apply padding to the input data
    x_padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    # Find the receptive field
                    roi = x_padded[n, i*stride:i*stride+FH, j*stride:j*stride+FW, :]
                    # Perform element-wise multiplication and summation
                    out[n, i, j, f] = np.sum(roi * w[f]) + b[f]
    
    return out
```

### Exercise 1.2: Efficient Forward Pass of Convolutional Layer

To improve computational efficiency, we optimized the forward pass by using more efficient matrix operations.

#### Code Snippets

```python
def efficient_forward(self, x, w, b, conv_param):
    N, H, W, C = x.shape
    F, FH, FW, _ = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    
    H_out = 1 + (H + 2 * pad - FH) // stride
    W_out = 1 + (W + 2 * pad - FW) // stride
    
    out = np.zeros((N, H_out, W_out, F))
    x_padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + FH
            w_start = j * stride
            w_end = w_start + FW
            x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]
            for f in range(F):
                out[:, i, j, f] = np.sum(x_slice * w[f], axis=(1, 2, 3))
    
    out += b.reshape((1, 1, 1, F))
    return out
```

## Exercise 2: Convolution Layer with pytorch
- [2023_SNU_DL_HW2_2.ipynb](https://github.com/jaewonlee16/Introduction-to-Deep-Learning/blob/master/HW2_student/2023_SNU_DL_HW2_2.ipynb)
- [HW_YourAnswer_cnn.py](https://github.com/jaewonlee16/Introduction-to-Deep-Learning/blob/master/HW2_student/HW_YourAnswer_cnn.py)
- This re-implements Exercise 1 using pytorch

## Conclusion

This readme covers the implementation details for the forward pass of a convolution layer in a CNN. 
The provided code snippets show both naive and efficient approaches to implementing the convolution operation. 
Further exercises involve implementing pooling layers and batch normalization, and training a CNN on the CIFAR-10 dataset.
