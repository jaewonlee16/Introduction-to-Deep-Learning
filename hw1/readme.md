# HW1 - Linear Classifier Implementation Using NumPy

---

## Overview

This project implements a linear classifier using NumPy modules. 
The primary goal is to understand the basic concepts of classification and learn how to implement forward and backward propagation for operations used in linear classifiers. 
The project includes the implementation of several key functions and classes and tests their performance on the CIFAR10 dataset.

### Exercise 1
In Exercise 1, classification on CIFAR10 dataset is implemented with Numpy (no pytorch).

[Notebook](https://github.com/jaewonlee16/Introduction-to-Deep-Learning/blob/master/hw1/2023_SNU_DL_HW1_1.ipynb)

 ### Exercise 2
In Exercise 2, classification on CIFAR10 dataset is implemented with Pytorch, which is much shorter.

[Notebook](https://github.com/jaewonlee16/Introduction-to-Deep-Learning/blob/master/hw1/2023_SNU_DL_HW1_2.ipynb)

### Implementations
Most of implementations are in [HW_YourAnswer.py](https://github.com/jaewonlee16/Introduction-to-Deep-Learning/blob/master/hw1/HW_YourAnswer.py)

## Repository Structure

- `HW_YourAnswer.py`: This file contains the implementations for the functions/classes required for the assignment.
- `utils.py`: This file contains utility functions used throughout the notebook.
- `2023_SNU_DL_HW1_1.ipynb`: Jupyter Notebook file for running the assignment and testing the functions implemented.
- `2023_SNU_DL_HW1_2.ipynb`: Jupyter Notebook file for running the assignment and testing the functions implemented.

## Key Components

1. **Softmax Function**
    - The softmax function is implemented to convert raw scores into probabilities.
    - Example output:
        ```python
        temp_x = np.array([[2060, 2000, 2080]])
        softmax_result1 = softmax(temp_x)
        print(softmax_result1) # [[2.06115362e-09 1.80485138e-35 9.99999998e-01]]
        ```

2. **Cross-Entropy Loss**
    - The cross-entropy loss is used to measure the performance of a classification model whose output is a probability value between 0 and 1.
    - Example output:
        ```python
        temp_score0 = np.array([[0.0, 0.0, 0.0]])
        temp_target0 = np.array([[0, 1, 0]])
        loss0 = cross_entropy_loss(temp_score0, temp_target0)
        print(loss0) # 20.72326583694641
        ```

3. **Linear Classifier**
    - Implemented linear classifier with functions to predict and compute cost.
    - Functions include `linear_predict`, `linear_cost_func`, `batch_gradient_descent_func`, and `stochastic_gradient_descent_func`.

4. **Gradient Descent**
    - Two methods of gradient descent are implemented:
        - Batch Gradient Descent
        - Stochastic Gradient Descent
    - These methods are used to optimize the weight parameters of the linear classifier.
  
5. **ReLU**
  - Forward propagation
  - Back propagation

6. **Sigmoid**
  - Forward propagation
  - Back propagation

7. **Affine**
  - Forward propagation
  - Back propagation

## Data Preprocessing

- **CIFAR10 Dataset**
    - The dataset consists of 50,000 training images and 10,000 test images. For this assignment, only 10,000 training images and 1,000 test images are used.
    - Data shapes:
        ```python
        # Training data
        Train data shape: (10000, 3072), Train labels shape: (10000,)
        # Test data
        Test data shape: (1000, 3072), Test labels shape: (1000,)
        ```

### Training and Evaluation

1. **Training the Model**
    - The linear classifier is trained on the CIFAR10 dataset using gradient descent methods.
    - Example training snippet:
        ```python
        initial_cost = linear_cost_func(x_batch, y_batch, W, b)
        print('Initial cost:', initial_cost) # 2.302585082994045
        ```

2. **Evaluation**
    - The model's performance is evaluated on the test dataset.
    - Test accuracy example:
        ```python
        test_acc = compute_accuracy(test_X, test_Y, W, b)
        print('Test accuracy:', test_acc) # 0.383
        ```

3. **Visualization**
    - Visualizes the learned weights for each class by rescaling the weight parameters.

## Instructions to Run

1. **Setup Environment**
    - Ensure you have Python and Jupyter Notebook installed.
    - Install necessary libraries:
        ```bash
        pip install numpy matplotlib torch
        ```

2. **Run the Notebook**
    - Open `2023_SNU_DL_HW1_1.ipynb` in Jupyter Notebook.
    - Execute cells sequentially to train and evaluate the model.

3. **Implement Functions**
    - Complete the functions in `HW_YourAnswer.py` as instructed in the notebook.
    - Re-run the notebook to test the implementations.

## Conclusion

This project demonstrates the implementation of a linear classifier from scratch using NumPy. 
By completing this assignment, you will gain a solid understanding of the classification process, data preprocessing, and the gradient descent optimization technique. 
The use of CIFAR10 dataset provides a practical example to apply these concepts in a real-world scenario.


---

