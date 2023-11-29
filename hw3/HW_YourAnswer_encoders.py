import torch
import torch.nn as nn
import torch.nn.functional as F

from HW_YourAnswer_modules import MultiLayerRNN, MultiLayerLSTM, MultiheadAttention

class RNNEncoder(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_dim, num_layers = 1, nonlinearity = 'tanh'):
        """
        Initialization of the RNNEncoder module.

        Args:
            num_tokens (int): Size of the vocabulary, representing the number of unique tokens.
            embedding_dim (int): Dimensionality of the embedding vectors for each token.
            hidden_dim (int): Size of the hidden state in the RNN layer.
            num_layers (int, optional): Number of RNN layers. Default is 1.
            nonlinearity (str, optional): Type of nonlinearity for the RNN layer.
                                          Can be 'tanh', 'relu', or 'sigmoid'. Default is 'tanh'.

        """
        super().__init__()

        self.embedding = None
        self.rnn = None
        self.linear_classifier = None

        # TODO: Initialize embedding layer, RNN layer, and linear classifier
        # ===================================================================================================== #
        # Details
        # - Initialize the embedding layer (self.embedding) with 'num_tokens' as the vocabulary size
        #   and 'embedding_dim' as the size of each embedding vector.
        # - Initialize the RNN layer (self.rnn) using the nn.RNN module with the following arguments:
        #   - input_size: embedding_dim
        #   - hidden_size: hidden_dim
        #   - num_layers: num_layers
        #   - nonlinearity: nonlinearity
        #   - batch_first: True, indicating the input shape as (batch_size, n_seq, input_size)
        # - Initialize the binary linear classifier layer (self.linear_classifier) with input size 'hidden_dim' and output size 2.
        # - Remember to set the batch_first argument to True when utilizing nn.RNN
        # ========================================== WRITE YOUR CODE ========================================== #


        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.rnn = nn.RNN(input_size = embedding_dim,
                          hidden_size = hidden_dim,
                          num_layers = num_layers,
                          nonlinearity = nonlinearity,
                          batch_first = True
                          )
        self.linear_classifier = nn.Linear(hidden_dim, 2)

        # ===================================================================================================== #

    def forward(self, text):
        """
        Forward pass of the RNNEncoder module.

        Args:
            text (torch.Tensor): Input tensor representing the input sequence of token indices.
                                Should have shape (batch_size, n_seq).

        Returns:
            torch.Tensor: Output tensor after passing through the RNN and linear classifier layers.
                          The output represents the final hidden state of the RNN and is reshaped
                          to match the linear classifier input shape.

        """
        # TODO: Forward pass
        # ===================================================================================================== #
        # Details
        # - Perform the forward pass by first embedding the input sequence using the embedding layer.
        # - Pass the embedded sequence through the RNN layer, obtaining the final hidden state.
        # - Use the final hidden state as input to the linear classifier layer and return the result.
        # ========================================== WRITE YOUR CODE ========================================== #


        embedded = self.embedding(text)
        output, _ = self.rnn(embedded)
        x = self.linear_classifier(output[:, -1, :])

        return x

        # ===================================================================================================== #

        return None

class LSTMEncoder(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_dim, num_layers):
        """
        Initialization of the LSTMEncoder module.

        Args:
            num_tokens (int): Size of the vocabulary, representing the number of unique tokens.
            embedding_dim (int): Dimensionality of the embedding vectors for each token.
            hidden_dim (int): Size of the hidden state in the LSTM layer.
            num_layers (int): Number of LSTM layers.

        """
        super().__init__()

        self.embedding = None
        self.lstm = None
        self.linear_classifier = None

        # TODO: Initialize embedding layer, LSTM layer, and linear classifier
        # ===================================================================================================== #
        # Details
        # - Initialize the embedding layer (self.embedding) with 'num_tokens' as the vocabulary size
        #   and 'embedding_dim' as the size of each embedding vector.
        # - Initialize the LSTM layer (self.lstm) using the nn.LSTM module with the following arguments:
        #   - input_size: embedding_dim
        #   - hidden_size: hidden_dim
        #   - num_layers: num_layers
        #   - batch_first: True, indicating the input shape as (batch_size, n_seq, input_size)
        # - Initialize the binary linear classifier layer (self.linear_classifier) with input size 'hidden_dim' and output size 2.
        # ========================================== WRITE YOUR CODE ========================================== #

        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim,
                          hidden_size = hidden_dim,
                          num_layers = num_layers,
                          batch_first = True
                          )
        self.linear_classifier = nn.Linear(hidden_dim, 2)

        # ===================================================================================================== #

    def forward(self, text):
        """
        Forward pass of the LSTMEncoder module.

        Args:
            text (torch.Tensor): Input tensor representing the input sequence of token indices.
                                Should have shape (batch_size, n_seq).

        Returns:
            torch.Tensor: Output tensor after passing through the LSTM and linear classifier layers.
                          The output represents the final hidden state of the LSTM and is reshaped
                          to match the linear classifier input shape.

        """
        # TODO: Forward pass
        # ===================================================================================================== #
        # Details
        # - Perform the forward pass by first embedding the input sequence using the embedding layer.
        # - Pass the embedded sequence through the LSTM layer, obtaining the final hidden state.
        # - Use the final hidden state as input to the linear classifier layer and return the result.
        # ========================================== WRITE YOUR CODE ========================================== #

        embedded = self.embedding(text)
        output, _ = self.lstm(embedded)
        x = self.linear_classifier(output[:, -1, :])

        return x
    
        # ===================================================================================================== #

        return None

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        """
        Initialization of FeedForward Class

        Args:
            embedding_dim (int): Dimensionality of the input feature vectors.

        """
        super().__init__()

        self.linear1 = None
        self.relu = None
        self.linear2 = None

        # TODO: Initialize linear layers
        # ===================================================================================================== #
        # Details
        # - Initialize the first linear layer (self.linear1) with input size 'embedding_dim' and output size '4 * embedding_dim'.
        # - Initialize the second linear layer (self.linear2) with input size '4 * embedding_dim' and output size 'embedding_dim'.
        # ========================================== WRITE YOUR CODE ========================================== #





        # ===================================================================================================== #

    def forward(self, x):
        """
        Forward pass of the FeedForward neural network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, n_seq, embedding_dim).

        Returns:
            torch.Tensor: Output tensor after passing through the linear layers and ReLU activation.

        """
        # TODO: Forward pass
        # ===================================================================================================== #
        # Details
        # - Pass the input tensor through the first linear layer.
        # - Apply the ReLU activation function.
        # - Pass the result through the second linear layer.
        # - Return the output tensor.
        # ========================================== WRITE YOUR CODE ========================================== #





    
        # ===================================================================================================== #

        return None

class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        """
        Initialization of EncoderBlock Class.

        Args:
            embedding_dim (int): Dimensionality of the input feature vectors.
            num_heads (int): Number of attention heads in the MultiheadAttention layer.

        """
        super().__init__()

        self.attention = None
        self.feedforward = None

        # TODO: Initialize attention and feedforward layers
        # ===================================================================================================== #
        # Details
        # - Initialize the attention layer (self.attention) with a MultiheadAttention module.
        #   Set the 'embed_dim' parameter to 'embedding_dim' and 'num_heads' parameter to 'num_heads'.
        # - Initialize the feedforward layer (self.feedforward) with a FeedForward module.
        #   Set the 'embedding_dim' parameter to 'embedding_dim'.
        # ========================================== WRITE YOUR CODE ========================================== #




        # ===================================================================================================== #

    def forward(self, x, mask = None):
        """
        Forward pass of the EncoderBlock.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor, optional): Attention mask tensor. Default is None.

        Returns:
            torch.Tensor: Output tensor after passing through attention and feedforward layers.

        """
        # TODO: Forward pass
        # ===================================================================================================== #
        # Details
        # - Pass the input tensor through the attention layer, obtaining the first residual (res1).
        # - Add res1 to the input tensor.
        # - Pass the result through the feedforward layer, obtaining the second residual (res2).
        # - Add res2 to the previous result.
        # - Return the final output tensor.
        # ========================================== WRITE YOUR CODE ========================================== #






    
        # ===================================================================================================== #

        return None

class TransformerEncoder(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_dim, num_layers = 1, max_pos_len = 2500):
        """
        Initialization of TransformerEncoder Class.

        Args:
            num_tokens (int): Size of the vocabulary, representing the number of unique tokens.
            embedding_dim (int): Dimensionality of the input feature vectors.
            hidden_dim (int): Size of the hidden state in the linear classifier layer.
            num_layers (int, optional): Number of EncoderBlocks in the Transformer Encoder. Default is 1.
            max_pos_len (int, optional): Maximum length of positional encoding. Default is 2500.

        """
        super().__init__()

        self.embedding = None
        self.encoders = None
        self.linear_classifier = None

        # TODO: Initialize embedding, encoder blocks, and linear classifier
        # ===================================================================================================== #
        # Details
        # - Initialize the embedding layer (self.embedding) with 'num_tokens' as the vocabulary size
        #   and 'embedding_dim' as the size of each embedding vector.
        # - Initialize the encoder blocks (self.encoders) using the nn.Sequential module with 'num_layers' blocks.
        # - Initialize the linear classifier layer (self.linear_classifier) with input size 'hidden_dim' and output size 2.
        # ========================================== WRITE YOUR CODE ========================================== #







        # ===================================================================================================== #

        pos_enc = None

        # TODO: Initialize positional encoding
        # ===================================================================================================== #
        # Details
        # - Create a tensor 'pos_enc' with zeros of shape (max_pos_len, embedding_dim).
        # - Calculate sinusoidal positional encoding and assign values to the tensor.
        # - Reshape 'pos_enc' and add a singleton dimension at the beginning for the broadcasting.
        # ========================================== WRITE YOUR CODE ========================================== #








        # ===================================================================================================== #

        self.register_buffer("pos_enc", pos_enc)

    def forward(self, text, pad_mask = None):
        """
        Forward pass of the TransformerEncoder.

        Args:
            text (torch.Tensor): Input tensor representing the input sequence of token indices.
                                Should have shape (batch_size, seq_len).
            pad_mask (torch.Tensor, optional): Padding mask tensor. Default is None.

        Returns:
            torch.Tensor: Output tensor after passing through encoder blocks and linear classifier layer.

        """
        # TODO: Forward pass
        # ===================================================================================================== #
        # Details
        # - Perform the forward pass by first embedding the input sequence using the embedding layer.
        # - Add the positional encoding to the embedded sequence.
        # - Pass the sequence through each encoder block in the 'encoders' sequential module.
        # - Extract the output from the first position in the sequence.
        # - Return the result after passing through the linear classifier layer.
        # ========================================== WRITE YOUR CODE ========================================== #









    
        # ===================================================================================================== #

        return None
