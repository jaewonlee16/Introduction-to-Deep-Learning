import numpy as np

from collections import OrderedDict
from utils import numerical_gradient

from collections import OrderedDict

# Exercise 1: Linear Classifier (Numpy)

def softmax(x):
    softmax_output = None
    # 1. Description
    # Implement softmax function that converts a vector of numbers into a vector of probabilities
    
    # ========================================== WRITE YOUR CODE ========================================== #
    # Instructions: 
    #     Implement the softmax function (w.r.t axis=1)
    #     1) Consider that the sum of the softmax output should be one
    #     NOTE : x can be vector or matrix and its sum of each row should be one with given matrix.
    #     2) Prevent the overflow (Do not let your function output the 'NaN')
    #     Check this article for overflow in softmax : https://medium.com/swlh/are-you-messing-with-me-softmax-84397b19f399
    max_input = np.max(x, axis=1).reshape(-1, 1)
    exponential = np.exp(x - max_input)
    row_sum = np.sum(exponential, axis = 1)
    softmax_output = (exponential.T / row_sum).T
    softmax_output[np.isnan(softmax_output)] = 0
    
    
    # ======================================================================================================
    return softmax_output    

def cross_entropy_loss(score, target):
    # 1. Description
    # Computes cross entropy loss using score, target
    
    delta = 1e-9
    batch_size = target.shape[0]
    CE_loss = 0
    
    # ========================================== WRITE YOUR CODE ========================================== #
    # Instructions : 
    #    - Use delta to prevent the occurence of log(0)
    #    - Return averaged loss w.r.t batch size(N)
    #    - NOTE : score, target can be vector or matrix.
    softmax_out = softmax(score)

    return -np.sum(target * np.log(score + delta)) / batch_size
        
    # ======================================================================================================
    
    

def linear_pred(X, weight, bias):
    # 1. Description
    # Computes probability score for given input X, weight, and bias
    
    # ========================================== WRITE YOUR CODE ========================================== #
    # Instructions : 
    #    - Implement linear classifier which returns softmax value of each class
    #    - Use 'softmax' fuction defined above
    #    - NOTE : bias is included in the weight
    
    batch_size = X.shape[0]
    bias_extend = np.tile(bias, (batch_size, 1))
    pred = X@weight + bias_extend
    
    # ======================================================================================================
    return softmax(pred)

def linear_cost_func(X, y, weight, bias):
    # 1. Description
    # Computes cross entropy loss of given X, y, weight, and bias
    
    # ========================================== WRITE YOUR CODE ========================================== #
    # Instructions : 
    #    - Implement linear classifier which returns softmax value of each class
    #    - Use fuctions defined above
    #    - NOTE : bias is included in the weight
    score = linear_pred(X, weight, bias)
    cost = cross_entropy_loss(score, y)
    # ======================================================================================================
    
    return cost
        
def batch_gradient_descent_func(X, y, weight, bias, alpha, num_iters):
    # 1. Description
    # Performs gradient descent to learn parameters
    # Updates parameters by taking num_iters gradient steps with learning rate alpha
    
    # 2. I/O info
    # Input
    #   1) X         : Data                              (N, D)
    #   2) y         : Label                             (N, )
    #   3) weight    : Initial model weight              (D, C)
    #   4) bias      : Initial model bias                (C, )
    #   4) alpha     : learning rate                     (scalar)
    #   5) num_iters : number of iterations for update
    # Output
    #   1) weight     : Final model weight                     (D, C)
    #   2) J_his     : history of cost per iteration             (num_iters, )
    #   3) W_his     : history of model weight per iteration  (num_iters, D, C)
    
    # (Notations)
    # N : number of data
    # D : data dimension (bias included)
    # C : number of classes
    
    J_his = np.zeros((num_iters,))
    W_his = np.zeros((num_iters,3072, 10))
    
    # You may use this variables
    n = len(X)
    
    for i in range(num_iters):
        W_his[i] = weight

        ### ========= YOUR CODE HERE ============
        # Instructions :
        #    - Perform a single gradient step on the parameter.
        #    - You should consider "vector multiplication" NOT loop statement.
        
        h = linear_pred(X, weight, bias)
        p = h - y
        gradient = X.T @ p
        weight = weight - alpha * gradient / n
        
        h = linear_pred(X, weight, bias)
        ### =====================================
        J_his[i] = cross_entropy_loss(h, y)

    return weight, bias, J_his, W_his
   
def stochastic_gradient_descent_func(X, y, weight, bias, alpha, num_iters, mini_batch, random_seed):
    # 1. Description
    # Performs gradient descent to learn parameters
    # Updates parameters by taking num_iters gradient steps with learning rate alpha
    
    # 2. I/O info
    # Same as 'batch_gradient_descent_func'
    
    # Initialize
    
    np.random.seed(random_seed)
    
    n = len(X)
    J_his = np.zeros((num_iters,))
    W_his = np.zeros((num_iters,3072, 10))
    for i in range(num_iters):
        W_his[i] = weight

        ### ========= YOUR CODE HERE ============
        # Instructions :
        #    - Perform a single gradient step on the parameter.
        #
        # 1) Randomly select a mini-batch from X, y
        #   - When you randomly sample, you should use 'np.random.choice' function for convenience of scoring.
        #   - This implementation assumes 'with-replacement' sampling.
        #   - Use variable 'mini-batch'
        # 2) With selected batch, calculate its gradient using "vector multiplication" NOT loop statement.
        
        random_index= np.random.choice(n, mini_batch, replace=True)
        x_random = X[random_index]
        y_random = y[random_index]

        h = linear_pred(x_random, weight, bias)
        p = h - y_random
        gradient = x_random.T @ p
        weight = weight - alpha * gradient / mini_batch
        

        ### =====================================
        J_his[i] = cross_entropy_loss(linear_pred(X, weight, bias), y)
    return weight, bias, J_his, W_his

# Exercise 2: Neural Network Modules (Numpy)

class OutputLayer:
    
    def __init__(self):
        self.loss = None           # loss value
        self.output_softmax = None # Output of softmax
        self.target_label = None   # Target label (one-hot vector)
        
    def forward(self, x, y):
        # 1. Description
        # Performs forward propagation of given x, y
    
        # 2. I/O info
        # Input
        #   1) X         : Data                                        (N, D)
        #   2) y         : Label                                       (N, )
        # Output
        #   1) self.loss : Cross entropy Loss(avg w.r.t batchsize)     (scalar)
    
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    - Compute cross entropy loss using given x, y
        #    - NOTE: Remember that the values in the forward propagation phase would be needed at the backward propagation phase)
               
        self.output_softmax = softmax(x)
        self.target_label = y
        self.loss = cross_entropy_loss(self.output_softmax, self.target_label)
        # ======================================================================================================
    
        return self.loss
    
    def backward(self, dout=1):
        # 1. Description
        # Performs back propagation of given x, y
    
        # 2. I/O info (Notations from Lecture 5 p.31-35)
        # Input
        #   1) dout   : dL/dL(=1)                                   (scalar)
        # Output
        #   1) ds     : dL/ds                                       (N, C)
        
        # You may use this variable
        batch_size = self.target_label.shape[0]
        
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    - Compute the backward propagation of the output layer
        #    - Since it is the output layer, the delta(dout) is one.
        #    - NOTE : ds should be divided by batch_size 
        #    - HINT : Calculate the derivative of the loss with respect to the 'softmax output') 
        
        ds = self.output_softmax - self.target_label
        # ======================================================================================================
        
        return ds / batch_size
    
    
class ReLU:
   
    def __init__(self):
        
        self.mask = None
        
    def forward(self, x):
        # 1. Description
        # Performs forward propagation of ReLU function
        
        # 2. I/O info
        # Input
        #   1) x       : input                                (any matrix or vector)
        # Output
        #   1) out     : output                               (same shape with x)
        
        self.out = None
    
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    - Implement ReLU function with given x (All the negative values should be ignored)
        #    - HINT : Think which value to save for the backward propagation phase
        self.mask = x > 0
        out = x * self.mask

        # ======================================================================================================
    
        return out
    
    def backward(self, dout):        
        # 1. Description
        # Performs back propagation of ReLU function
        
        # 2. I/O info
        # Input
        #   1) dout   : propagation value from the upper layer       (same shape with x)
        # Output
        #   1) dx     : propagation value from ReLU                  (same shape with x)
    
        dx = None
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    - dout is the propagation value from the upper layer
        #    - Only the points that had survived during the forward propagation should be backward propagated
        dx = dout * self.mask

        # ======================================================================================================
        
        return dx
    
class Sigmoid:
    
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        # 1. Description
        # Performs forward propagation of Sigmoid function
        
        # 2. I/O info
        # Input
        #   1) x       : input                                (any matrix or vector)
        # Output
        #   1) out     : output                               (same shape with x)
        
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    - Implement sigmoid function
        #    - Make sure that the output is in the range of 0 to 1
        self.out = 1 / (1 + np.exp(-x))

        # ======================================================================================================
    
        return self.out
    
    def backward(self, dout):
        # 1. Description
        # Performs back propagation of Sigmoid function
        
        # 2. I/O info
        # Input
        #   1) dout   : propagation value from the upper layer       (same shape with x)
        # Output
        #   1) dx     : propagation value from Sigmoid               (same shape with x)
        
        dx = None
        
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    - dout is the propagation value from the upper layer
        #    - Consider the derivative of the sigmoid function
        dx = self.out * (1 - self.out) * dout
        # ======================================================================================================
        
        return dx
    
class Affine:
    
    def __init__(self, W, b):
        
        # : bias is considered seperately !!
        
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        # 1. Description
        # Performs forward propagation of Affine function
        
        # 2. I/O info
        # Input
        #   1) x       : input                                (N, D_1)
        #   2) W       : weight parameters                    (D_1, D_2)
        # Output
        #   1) out     : output                               (N, D_2)
        
        out = None
    
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    - Implement an Affine layer
        #    - NOTE : bias should be considered separately ! (Not included in self.W)
        #    - Consider the backward propagation phase
        self.x = x
        out = self.x @ self.W + np.tile(self.b, (self.x.shape[0], 1))
        
        # ======================================================================================================
    
        return out
    
    def backward(self, dout):
        # 1. Description
        # Performs back propagation of Affine function
        
        # 2. I/O info
        # Input
        #   1) dout     : propagation value from the upper layer      (N, D_2)                                
        # Output
        #   1) dx       : propagation value from Affine function      (N, D_1)
        
        dx = None
        
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    - Compute the back-propagation value with respect to input x, weight W and bias b
        #    - Do not use loop statement
        #    - NOTE : You should also compute the derivative dW(self.dW) and db(self.db)
        #             (dW and db will be used when weights are updated)
        #             - dW : (D_1, D_2)
        #             - db : (1, D_2)
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis = 0)
        

        # ======================================================================================================
        
        return dx
    

class TwoLayerNet:
    # 1. Description
    # Implement forward/back propagation of 'TwoLayerNet' using classes defined above
    # Brief information of the functions are written below.
    # Please review Lecture 6 p.22 - 26
    
    # NOTE: L2-regularization will be also used to prevent overfitting
    # 
    # 1) __init__()
    #   A function that initialize Weight and bias
    #    
    # 2) predict()
    #   A function that performs forward propagation to return output score
    #    
    # 3) loss()
    #   A function that computes the loss using the forward propagation results with respect to the input
    #    
    # 4)accuracy()
    #   A function that computes the accuracy using the Output data and True label
    #    
    # 5) gradient()
    #   A function that performs backward propagation of Neural network using the input and tht true label
    #
    # 6) update_weight() 

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, regularization = 0.0):
        # 1. I/O info
        # Input
        #   1) input_size      : size of input dimension            (D_1)                                           
        #   2) hidden_size     : size of hidden dimension           (D_2)
        #   3) output_size     : size of output dimension           (C)
        #   4) weight_init_std  : std used for weight initialization
        #   5) regularization  : regularization strength (lambda in Lecture 4 p.19)
        
        # Weight Initialization
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        self.reg = regularization

        self.layers = OrderedDict()
        
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #     - Implement two layers net by adding layers in 'self.layers' dictionary
        #       (ex) self.layers['affine'] = ~~
        #     - Use the classes(Affine, ReLU, OutputLayer) defined above
        #     - NOTE: OrderedDict() remembers the order entries were added
        #     - Model Structure would be like
        #       " Input => Fully Connected => ReLU => Fully Connected => OutputLayer "
        self.layers['FC1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU'] = ReLU()
        self.layers['FC2'] = Affine(self.params['W2'], self.params['b2'])
        

        # ======================================================================================================
        
        # NOTE : OutputLayer (softmax & crossentropy loss) is defined seperately
        self.lastLayer = OutputLayer()

    def predict(self, x):
        # 1. I/O info
        # Input
        #   1) x      : input data      (N, D_1)
        # Output
        #   2) x      : output score    (N, C)
        
        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #     - Forward propagate TwolayerNet to return output score (not probability)
        #     - HINT : the code only requires 2 lines
        x = self.layers['FC2'].forward(self.layers['ReLU'].forward(self.layers['FC1'].forward(x)))
        #x = self.lastLayer.forward(x, )
        x = softmax(x)

        
        # ======================================================================================================    
        
        return x

    def loss(self, x, y):
        # 1. I/O info
        # Input
        #   1) x      : input data                         (N, D_1)
        #   2) y      : label                              (N, C)
        # Output
        #   2) loss   : cross entropy loss + regularization loss of given x,y    (scalar)

        # ========================================== WRITE YOUR CODE ========================================== #
        # Instructions :
        #    Implement function loss that computes the cross entropy loss and regularizaiton loss
        #
        #    1) Cross Entropy
        #       - Compute score of given x and 
        #       - Hint: use predict() function
        #    2) Regularization
        #       - Implement L2 Regularization if self.reg is not 0
        #         (Multiply 0.5 to the reg_loss for computational convenience)
        #       - Use 'self.reg' as a regularization constant
        weight_squared = np.power(self.params['W1'], 2).sum() + np.power(self.params['W2'], 2).sum() 

        loss = self.lastLayer.forward(self.layers['FC2'].forward(self.layers['ReLU'].forward(self.layers['FC1'].forward(x))), y) + self.reg * 0.5 * weight_squared

        # ===================================================================================================== #             
        return loss
    

    def accuracy(self, x, y):

        score = self.predict(x)
        score = np.argmax(score, axis=1)
        if y.ndim != 1 : y = np.argmax(y, axis=1)
        accuracy = np.sum(score == y) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, y):

        loss_W = lambda W: self.loss(x, y)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        if self.reg != 0.0:
            pass
            
        return grads

    def gradient(self, x, y):
        
        self.grads = {}
        
        # 1. I/O info
        # Input
        #   1) x      : input data                         (N, D_1)
        #   2) y      : label                              (N, C)  
        # Output
        #   1) self.grads  : dictionary that has key, value pair 
        #               for the name of the parameters and the gradient value
        #               (ex) self.grads['W1'] = dL/dW1          (D_1, D_2)

        # [STEP 1] forward propagation
        _ = self.loss(x, y)

        # [STEP 2] backward propagation & save gradients for each params in 'grads'
    
        # # ========================================== WRITE YOUR CODE ========================================== #
    
        # Instructions :
        #     - Implement 1) back-propagation 2) add (param name, param gradient) pair in grads
        
        #     1) back-propagation
        #       - Implement backpropagation looping all the layers in 'self.layers'
        #       - NOTE: the propagation should proceed in reverse order
        #               (Outputlayer -> Affine2 -> ReLU -> Affine1)
        #       - HINT: list has in built function 'reverse'
        #     2) add (param name, param gradient) pair in grads
        #       - save the computed gradient values of each layer during backpropagation
        #       - gradients of L2 regularization should also be considered !
        second2last_grad = self.layers['ReLU'].backward(self.layers['FC2'].backward(self.lastLayer.backward()))
        self.grads['W2'] = self.reg * self.params['W2'] + self.layers['FC2'].dW
        self.grads['b2'] = self.layers['FC2'].db

        _ = self.layers['FC1'].backward(second2last_grad)
        self.grads['W1'] = self.reg * self.params['W1'] + self.layers['FC1'].dW
        self.grads['b1'] = self.layers['FC1'].db
        

        
        # ===================================================================================================== #

        return self.grads
    
    def update_params(self, learning_rate):
        
        # 1. I/O info
        # Input
        #   1) learning_rate      : learning rate for gradient descent         (scalar)
        
        # # ========================================== WRITE YOUR CODE ========================================== #
    
        # Instructions :
        #     - Update weights by gradient descent
        for key in self.params:
            self.params[key] = self.params[key] - learning_rate * self.grads[key]
        print(f"{self.params['W1'] = }")    
        # ===================================================================================================== #
