import torch
import torch.nn as nn

import numpy as np

class OneLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity = "tanh"):
        """
        Initialization of the OneLayerRNN module.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state.
            nonlinearity (str, optional): Activation function for the RNN. Default is "tanh".

        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Set the activation function based on the provided nonlinearity argument
        if nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        elif nonlinearity == "sigmoid":
            self.nonlinearity = torch.sigmoid
        elif nonlinearity == "relu":
            self.nonlinearity = torch.relu

        self.ih_fc = None
        self.hh_fc = None

        # TODO: Initialize linear layers
        # ===================================================================================================== #
        # Details
        # - Initialize the input-to-hidden linear layer (ih_fc) with input_size and hidden_size.
        #   This layer maps input features to the hidden state.
        # - Initialize the hidden-to-hidden linear layer (hh_fc) with hidden_size for both input and output.
        #   This layer captures the recurrence in the RNN by mapping the current hidden state to the next one.
        # ========================================== WRITE YOUR CODE ========================================== #
        self.ih_fc = nn.Linear(self.input_size, self.hidden_size)
        self.hh_fc = nn.Linear(self.hidden_size, self.hidden_size)




        # ===================================================================================================== #
    
    def forward(self, x_seq, h_init = None):
        """
        Forward pass of the OneLayerRNN.

        Args:
            x_seq (torch.Tensor): Input sequence of shape (batch_size, n_seq, input_size).
            h_init (torch.Tensor, optional): Initial hidden state. Default is None.

        Returns:
            torch.Tensor: Output sequence of hidden states of shape (batch_size, n_seq, hidden_size).
            torch.Tensor: Final hidden state of shape (batch_size, hidden_size).

        """
        n_batch, n_seq, _ = x_seq.shape

        # If an initial hidden state is provided, use it; otherwise, initialize with zeros.
        if h_init is not None:
            h = h_init.clone()
        else:
            h = torch.zeros((n_batch, self.hidden_size)).to(x_seq)
        
      	# TODO: Forward pass of the OneLayerRNN
        # ===================================================================================================== #
        # Details:
        # - Iterate through the input sequence to compute the hidden states at each time step.
        # - The loop applies the RNN update at each time step using the input and previous hidden state.
        # - The resulting hidden state is stored in the 'output' list for each time step.
        # - The final hidden state 'h' is updated after each time step.
        # - 'output' is stacked along the sequence dimension to form the output sequence.
        # ========================================== WRITE YOUR CODE ========================================== #
        output = torch.zeros([n_batch, n_seq, self.input_size]).to(x_seq)
        h_t = h
        for t in range(n_seq):
            x_t = x_seq[:, t, :]
            h_t = self.nonlinearity(self.ih_fc(x_t) + self.hh_fc(h_t))
            output[:, t, :] = h_t

        return output, h_t









    
        # ===================================================================================================== #

class MultiLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity = "tanh"):
        """
        Initialization of the MultiLayerRNN module.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state.
            num_layers (int): Number of layers in the RNN.
            nonlinearity (str, optional): Activation function for the RNN. Default is "tanh".

        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = None

      	# TODO: Initialize RNN layers
        # ===================================================================================================== #
        # Details:
        # - Initialize a list 'rnn' containing OneLayerRNN instances.
        # - The list is then passed to nn.Sequential to create a sequential RNN model.
        # ========================================== WRITE YOUR CODE ========================================== #
        layers = []
        for _ in range(self.num_layers):
            l = OneLayerRNN(input_size = self.input_size, hidden_size = self.hidden_size, nonlinearity = nonlinearity)
            layers.append(l)

        self.rnn = nn.Sequential(*layers)







        # ===================================================================================================== #

    
    def forward(self, x_seq, h_init = None):
        """
        Forward pass of the MultiLayerRNN module.

        Args:
            x_seq (torch.Tensor): Input sequence of shape (batch_size, n_seq, input_size).
            h_init (list of torch.Tensor, optional): Initial hidden states for each layer.
                Default is None, indicating initialization with zeros.

        Returns:
            torch.Tensor: Output sequence of hidden states of shape (batch_size, n_seq, hidden_size).
            torch.Tensor: Final hidden states for each layer of shape (num_layers, batch_size, hidden_size).

        """
        n_batch = x_seq.shape[0]

        # TODO: Forward Pass
        # ===================================================================================================== #
        # Details:
        # - Clone the input sequence to avoid in-place modification.
        # - Initialize a list 'h_lst' to store the final hidden state for each layer.
        # - Iterate through each layer of the RNN, updating the input sequence and collecting final hidden states.
        # - If initial hidden states are provided, use them; otherwise, initialize with zeros.
        # - The loop updates 'out' and 'h_last' at each layer.
        # - 'out' represents the output sequence, and 'h_last' is the final hidden state of the current layer.
        # - 'h_lst' is a list of final hidden states for all layers, and it is stacked along the layer dimension.
        # ========================================== WRITE YOUR CODE ========================================== #


        out = x_seq
        h_last = torch.zeros([self.num_layers, n_batch, self.hidden_size]).to(x_seq)

        for i_layer in range(self.num_layers):
            h_init_layer = h_init[i_layer] if h_init is not None else None
            out, h_last[i_layer, :, :] = self.rnn[i_layer](out, h_init_layer)


        return out, h_last




    
        # ===================================================================================================== #
                
class OneLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Initialization of the OneLayerLSTM module.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state.

        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ih_fc = None
        self.hh_fc = None

        # TODO: Initialize linear layers for LSTM
        # ===================================================================================================== #
        # Details:
        # - Initialize the input-to-hidden linear layer (ih_fc) with input_size and hidden_size * 4.
        #   This layer maps input features to the concatenated input for the LSTM gates.
        # - Initialize the hidden-to-hidden linear layer (hh_fc) with hidden_size for both input and output,
        #   and multiplied by 4 to match the input size for the concatenated input of LSTM gates.
        #   This layer captures the recurrence in the LSTM by mapping the current hidden state to the next one.
        # ========================================== WRITE YOUR CODE ========================================== #




        # ===================================================================================================== #

    def linear(self, x, h):
        """
        Compute the linear transformation for the LSTM gates.

        Args:
            x (torch.Tensor): Input tensor.
            h (torch.Tensor): Hidden state tensor.

        Returns:
            torch.Tensor: Tensors corresponding to the input gate.
            torch.Tensor: Tensors corresponding to the forget gate.
            torch.Tensor: Tensors corresponding to the cell gate.
            torch.Tensor: Tensors corresponding to the output gate.

        """
        hidden_size = self.hidden_size

        ih = self.ih_fc(x)
        hh = self.hh_fc(h)

        i, f, g, o = None, None, None, None

        # TODO: Compute the linear transformation for LSTM gates
        # ===================================================================================================== #
        # Details:
        # - Split the concatenated input for the LSTM gates.
        # - Apply activation functions to each gate: sigmoid for i, f, and o, and tanh for g.
        # - These gates control the flow of information in the LSTM cell.
        # ========================================== WRITE YOUR CODE ========================================== #






        # ===================================================================================================== #

        return i, f, g, o
    
    def forward(self, x_seq, init = None):
        """
        Forward pass of the OneLayerLSTM.

        Args:
            x_seq (torch.Tensor): Input sequence of shape (batch_size, n_seq, input_size).
            init (tuple of torch.Tensor, optional): Initial hidden and cell states.
                Default is None, indicating initialization with zeros.

        Returns:
            torch.Tensor: Output sequence of hidden states of shape (batch_size, n_seq, hidden_size).
            (torch.Tensor, torch.Tensor): Tuple of final hidden and cell states each of shape (batch_size, hidden_size).

        """
        n_batch, n_seq, _ = x_seq.shape

        # If initial hidden and cell states are provided, use them; otherwise, initialize with zeros.
        if init is not None:
            h = init[0].clone()
            c = init[1].clone()
        else:
            h = torch.zeros((n_batch, self.hidden_size)).to(x_seq)
            c = torch.zeros((n_batch, self.hidden_size)).to(x_seq)

        # TODO: Forward pass
        # ===================================================================================================== #
        # Details:
        # - Iterate through the input sequence to compute the hidden states at each time step.
        # - The loop applies the LSTM update at each time step using the input and previous hidden and cell states.
        # - The resulting hidden state is stored in the 'output' list for each time step.
        # - The final hidden and cell states 'h' and 'c' are updated after each time step.
        # - 'output' is stacked along the sequence dimension to form the output sequence.
        # ========================================== WRITE YOUR CODE ========================================== #
        








    
        # ===================================================================================================== #


class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        Initialization of the MultiLayerLSTM module.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state.
            num_layers (int): Number of layers in the LSTM.

        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = None

        # TODO: Initialize LSTM layers
        # ===================================================================================================== #
        # Details:
        # - Initialize a list 'lstm' containing OneLayerLSTM instances.
        # - The list is then passed to nn.Sequential to create a sequential LSTM model.
        # ========================================== WRITE YOUR CODE ========================================== #






        # ===================================================================================================== #
    
    def forward(self, x_seq, init = None):
        """
        Forward pass of the MultiLayerLSTM.

        Args:
            x_seq (torch.Tensor): Input sequence of shape (batch_size, n_seq, input_size).
            init (tuple of torch.Tensor, optional): Initial hidden and cell states.
                Default is None, indicating initialization with zeros.

        Returns:
            torch.Tensor: Output sequence of hidden states of shape (batch_size, n_seq, hidden_size).
            (torch.Tensor, torch.Tensor): Tuple of final hidden and cell states each of shape (num_layers, batch_size, hidden_size).

        """
        n_batch = x_seq.shape[0]

        # TODO: Forward pass of the MultiLayerLSTM
        # ===================================================================================================== #
        # Details:
        # - Clone the input sequence to avoid in-place modification.
        # - Initialize lists 'h_lst' and 'c_lst' to store the final hidden and cell states for each layer.
        # - Iterate through each layer of the LSTM, updating the input sequence and collecting final hidden and cell states.
        # - If initial hidden and cell states are provided, use them; otherwise, initialize with zeros.
        # - The loop updates 'out', 'h_last', and 'c_last' at each layer.
        # - 'out' represents the output sequence, and 'h_last' and 'c_last' are the final hidden and cell states of the current layer.
        # - 'h_lst' and 'c_lst' are lists of final hidden and cell states for all layers, and they are stacked along the layer dimension.
        # ========================================== WRITE YOUR CODE ========================================== #














    
        # ===================================================================================================== #

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim = None, vdim = None):
        """
        Initialization of the MultiheadAttention module.

        Args:
            embed_dim (int): Dimensionality of the input and output embeddings.
            num_heads (int): Number of attention heads.
            kdim (int, optional): Dimensionality of the key projections. Default is None, indicating use of embed_dim.
            vdim (int, optional): Dimensionality of the value projections. Default is None, indicating use of embed_dim.

        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Set key and value dimensions, defaulting to embed_dim if not provided.
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None

        # TODO: Initialize linear projections for query, key, value, and output.
        # ===================================================================================================== #
        # Details
        # - Initialize the query projection (q_proj) with input embed_dim.
        # - Initialize the key projection (k_proj) with kdim.
        # - Initialize the value projection (v_proj) with vdim.
        # - Initialize the output projection (out_proj) with output embed_dim.
        # ========================================== WRITE YOUR CODE ========================================== #






        # ===================================================================================================== #

        # Set all bias to be zero vectors
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def split_heads(self, x):
        """
        Split the input tensor into multiple heads.

        Args:
            x (torch.Tensor): Input tensor of shape (B, -1, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_heads, -1, embed_dim // num_heads).

        """
        n_batch = x.shape[0]

        # TODO: Reshape the tensor to split into heads.
        # ===================================================================================================== #
        # Details
        # - Reshape the input tensor x to have shape (B, -1, num_heads, embed_dim // num_heads).
        # - Transpose the dimensions to get the desired shape (B, num_heads, -1, embed_dim // num_heads).
        # ========================================== WRITE YOUR CODE ========================================== #



        # ===================================================================================================== #
    
    def scaled_dot_product_attention(self, wq, wk, wv, pad_mask = None):
        """
        Scaled Dot-Product Attention mechanism.

        Args:
            wq (torch.Tensor): Query tensor after linear projection (B, num_heads, n_seq, embed_dim // num_heads).
            wk (torch.Tensor): Key tensor after linear projection (B, num_heads, n_key, embed_dim // num_heads).
            wv (torch.Tensor): Value tensor after linear projection (B, num_heads, n_key, embed_dim // num_heads).
            pad_mask (torch.Tensor, optional): Padding mask tensor of shape (B, n_key) indicating positions to be masked. Default is None.

        Returns:
            torch.Tensor: Scaled Dot-Product Attention output tensor reshaped to (B, n_seq, embed_dim).
            torch.Tensor: Average attention weights across heads for visualization (B, n_seq, n_key).

        """
        
        n_batch = wq.shape[0]
        d_k = self.embed_dim // self.num_heads

        # TODO: Compute attention scores.
        # ===================================================================================================== #
        # Details
        # - Transpose the key tensor wk to shape (B, num_heads, embed_dim // num_heads, n_key).
        # - Perform matrix multiplication between query wq and transposed key wk.
        # - Scale the result by the square root of d_k.
        # - Apply the padding mask if provided.
        # - Apply the softmax activation along the key dimension.
        # - Calculate the average attention weight.
        # - Perform matrix multiplication between the attention scores and value tensor wv.
        # - Transpose the result to the original shape.
        # ========================================== WRITE YOUR CODE ========================================== #

        """
        Refer to the following notes if you think you need them:
        wq, wk, wv: query, key, value after linear projection W
        wq: (B, num_heads, n_seq, embed_dim // num_heads)
        wk: (B, num_heads, n_key, embed_dim // num_heads)
        wv: (B, num_heads, n_key, embed_dim // num_heads)
        wq @ wk.T: (B, num_heads, n_seq, n_key)
        pad_mask: (B, n_key)
        softmax @ v: (B, num_heads, n_seq, embed_dim // num_heads)
        """










    
        # ===================================================================================================== #
    
    def forward(self, q, k, v, pad_mask = None):
        """
        Forward pass of the MultiheadAttention module.

        Args:
            q (torch.Tensor): Query tensor of shape (B, n_seq, embed_dim).
            k (torch.Tensor): Key tensor of shape (B, n_key, kdim).
            v (torch.Tensor): Value tensor of shape (B, n_key, vdim).
            pad_mask (torch.Tensor, optional): Padding mask tensor indicating positions to be masked. Default is None.

        Returns:
            torch.Tensor: Output tensor after the Multihead Attention mechanism, reshaped to (B, -1, embed_dim).
            torch.Tensor: Average attention weights across heads for visualization of shape (B, n_seq).

        """

        # TODO: Forward Pass.
        # ===================================================================================================== #
        # Details
        # - Linearly project the query, key, value tensors using the linear projection layers.
        # - Use the scaled_dot_product_attention method to perform scaled dot-product attention.
        # - Pass the linearly projected query 'wq', key 'wk', and value 'wv' tensors along with the padding mask.
        # - The result 'x' is the output tensor after attention.
        # - The 'weight' tensor represents the average attention weights across heads for visualization.
        # - Linearly project the output tensor 'x' using the linear projection layer 'self.out_proj'.
        # ========================================== WRITE YOUR CODE ========================================== #
        """
        Refer to the following notes if you think you need them:
        q: (B, n_seq, embed_dim)
        k: (B, n_key, kdim)
        v: (B, n_key, vdim)
        pad_mask: (B, n_key)
        """






    
        # ===================================================================================================== #

        return None
