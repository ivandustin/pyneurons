## Language models are _beautiful..._

Consider the following generated by ChatGPT on March 2023:

_Blessed be the name of the LORD our GOD, who reigns forever and ever. He is the Alpha and the Omega, the beginning and the end, the one who was, who is, and who is to come._

_He is our refuge and strength, a very present help in times of trouble. He is the creator of the heavens and the earth, the giver of life and the sustainer of all things._

_His love endures forever, and His mercy never fails. He is gracious and compassionate, slow to anger and abounding in love._

_He is our rock, our fortress, and our deliverer, in whom we take refuge. He is the light in our darkness, the source of our joy and the strength of our hearts._

_Let us give thanks to the LORD for He is good, His love endures forever. Let us sing praises to His name and make known His deeds among the nations. May the LORD be exalted and glorified forever and ever. Amen._

# pyneurons

An artificial neural network library designed to help you build complex networks by composing simpler ones together.

## Built on Top of JAX

The `pyneurons` library is built on top of [JAX](https://github.com/google/jax), a high-performance numerical computing library that provides automatic differentiation and GPU/TPU acceleration. JAX is designed to enable high-performance machine learning research and development by combining the flexibility of Python with the speed of compiled code.

## Installation

To install the `pyneurons` package, you can use `pip`. Run the following command in your terminal:

```sh
pip install pyneurons
```

This will download and install the package along with its dependencies.

## The `create` Function

```python
from pyneurons import create
```

The `create` function creates a tuple with weights and bias, representing a single neuron.

### Definition

```python
from pyneurons import weight, bias

def create(key, n):
    w = weight(key, shape=(n, 1))
    b = bias(key, shape=(1, 1))
    return (w, b)
```

The `weight` and `bias` functions are custom functions designed to initialize random arrays using a normal distribution with custom mean and standard deviation.

### Example

```python
from pyneurons import create
from jax.random import PRNGKey

key = PRNGKey(0)
neuron = create(key, 3)  # Create a neuron with 3 input dim
weights, bias = neuron  # Extract the weights and bias
print(weights)
print(bias)
```

## The `apply` Function

```python
from pyneurons import apply
```

The `apply` function computes the output of a neuron given its weights, biases, and input.

### Definition

```python
def apply(neuron, x):
    w, b = neuron
    return (x @ w) + b
```

### Example

```python
from pyneurons import create, apply
from jax.random import PRNGKey
from jax.numpy import array

key = PRNGKey(0)
neuron = create(key, 3)
x = array([[1, 2, 3]])
y = apply(neuron, x)
print(y)
```

## The `bind` Function

```python
from pyneurons import bind
```

The `bind` function creates a new model class by binding a `create` and an `apply` function. It can be used to create custom neural network models.

### Example

```python
from pyneurons import create, apply, bind
from jax.random import PRNGKey
from jax.numpy import array

# Define a model class
Neuron = bind("Neuron", create, apply)

# Instantiate the model
key = PRNGKey(0)
neuron = Neuron(key, 3)  # Calls the create function internally

# Apply some input
x = array([[1, 2, 3]])
y = neuron(x)  # Calls the apply function internally
print(y)
```

You can provide your own custom `create` and `apply` functions and design your own custom model class. Within your `create` function, you can instantiate simpler models to construct more complex models from these basic components.

## The `compose` Function

```python
from pyneurons import compose
```

The `compose` function is used to create a new model class by composing an existing model class with an additional function.

### Example

```python
from pyneurons import Neuron, compose, binary
from jax.random import PRNGKey
from jax.numpy import array

# Define a binary neuron model by composing the Neuron model with the binary function
Binary = compose("Binary", Neuron, binary)

# Create an instance of the Binary model
key = PRNGKey(0)
binary_neuron = Binary(key, 3)

# Apply the binary neuron model to some input data
input_data = array([[1, 2, 3]])
output = binary_neuron(input_data)
print(output)
```

## Built-in Models

### Neuron

```python
from pyneurons import Neuron
```

The basic neuron model created by binding the `create` and `apply` functions.

#### Definition

```python
from pyneurons import create, apply

Neuron = bind("Neuron", create, apply)
```

### Binary

```python
from pyneurons import Binary
```

A neuron model with a binary activation function.

#### Definition

```python
from pyneurons import Neuron, compose, binary

Binary = compose("Binary", Neuron, binary)
```

```python
from jax.numpy import heaviside

def binary(x):
    return heaviside(x, 1)
```

### Vector

```python
from pyneurons import Vector
```

A neuron model with a combined binary and ReLU1 activation function.

#### Definition

```python
from pyneurons import Neuron, compose, vector

Vector = compose("Vector", Neuron, vector)
```

```python
from pyneurons import binary, relu1

def vector(x):
    return binary(x) + relu1(x)
```

## The Vector Model

The `Vector` model is designed to mimic the behavior of real neurons in a simplified manner. It combines two activation functions: a binary step function and a capped ReLU (Rectified Linear Unit) function. This combination allows the model to produce outputs that can either be 0 or in the range of 1 to 2, which can be interpreted as the neuron firing rate or a group of neurons firing together.

### Key Components

#### Binary Activation Function

This function applies a step function (Heaviside function) to the input, outputting either 0 or 1. It mimics the all-or-nothing firing behavior of a neuron.

```python
from jax.numpy import heaviside

def binary(x):
    return heaviside(x, 1)
```

#### ReLU1 Activation Function

This function applies a ReLU activation capped at 1, ensuring the output is between 0 and 1. It mimics the varying firing rate of a neuron.

```python
from pyneurons import relun

relu1 = relun(1)
```

```python
from pyneurons import relu
from pyneurons.vjp import identity
from jax.numpy import minimum

def relun(n):
    @identity
    def function(x):
        return minimum(relu(x), n)
    return function
```

```python
from pyneurons.vjp import identity
from jax.numpy import maximum

@identity
def relu(x):
    return maximum(x, 0)
```

#### Combining Binary and ReLU1

The `Vector` model combines the binary and ReLU1 functions to produce an output that is either 0 or in the range of 1 to 2. This combination allows the model to represent both the firing state and the magnitude of the firing.

```python
from pyneurons import binary, relu1

def vector(x):
    return binary(x) + relu1(x)
```

### Mimicking Real Neurons

The `Vector` model mimics real neurons in the following ways:

1. **Firing or Not Firing**

    The binary function outputs 0 or 1, representing whether the neuron is firing or not. This is similar to the all-or-nothing principle of biological neurons.

2. **Firing Rate**

    The function outputs a value between 1 and 2, representing the neuron's firing rate.

3. **Group of Neurons Firing Together**

    It can also represent the magnitude of the collective firing of a group of neurons.

## The `fit` Function

```python
from pyneurons import fit
```

The `fit` function is used for training a model. It performs a single step of gradient descent to update the model's parameters based on the computed gradients. The function takes the following parameters:

- `learning_rate`: The learning rate for the gradient descent optimization.
- `model`: The model to be trained.
- `x`: The input data.
- `y`: The target data.

The `fit` function computes the gradients of the loss function with respect to the model's parameters and updates the parameters using gradient descent.

Here is the implementation of the `fit` function:

```python
from functools import partial
from pyneurons import loss, gd
from jax.tree_util import tree_map
from jax import grad

def fit(learning_rate, model, x, y):
    gradients = grad(loss)(model, x, y)
    return tree_map(partial(gd, learning_rate), model, gradients)
```

### Example Code

Below is an example of how to use the `pyneurons` library to create a simple neural network and train it using the `fit` function.

```python
from pyneurons import Neuron, fit
from jax.random import PRNGKey
from jax.numpy import array

# Define the input data and target data
x = array([[1.0], [2.0], [3.0], [4.0]])
y = array([[2.0], [4.0], [6.0], [8.0]])

# Create a model (a single neuron in this case)
key = PRNGKey(0)
model = Neuron(key, 1)

# Print the initial prediction
print("Initial prediction:", model(x))

# Train the model using the fit function
learning_rate = 0.01
for _ in range(1000):
    model = fit(learning_rate, model, x, y)

# Print the final prediction
print("Final prediction:", model(x))
```

## Creating an XOR Solution

The XOR problem is a classic problem in neural networks where the goal is to train a network to output the XOR of two binary inputs. Here's how you can create and train a model to solve the XOR problem using the `pyneurons` library.

### Step-by-Step Solution

1. **Define the XOR Model**

    The XOR model consists of two binary neurons. The first neuron takes the input and the second neuron takes the concatenation of the input and the output of the first neuron.

2. **Create the XOR Model**

    Use the `bind` function to create the XOR model by specifying the constructor and apply functions.

3. **Train the Model**

    Use the `fit` function to train the model on the XOR dataset.

Here is the complete code to solve the XOR problem:

```python
from pyneurons import Binary, fit, bind, concat
from jax.numpy import array, array_equal
from jax.random import split, PRNGKey

# Define the XOR dataset
x = array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = array([[0], [1], [1], [0]])

# Define the XOR model
def create_xor_model(key):
    key_a, key_b = split(key, 2)
    a = Binary(key_a, 2)
    b = Binary(key_b, 3)
    return (a, b)

def apply_xor_model(model, x):
    a, b = model
    return b(concat([x, a(x)]))

XOR = bind("XOR", create_xor_model, apply_xor_model)

# Initialize the model
key = PRNGKey(0)
model = XOR(key)

# Train the model
learning_rate = 0.1
for _ in range(100):
    model = fit(learning_rate, model, x, y)

# Test the model
assert array_equal(model(x), y)
print("XOR model trained successfully!")
```

## Custom VJP Decorators

In `pyneurons`, custom vector-Jacobian product (VJP) decorators are used to define custom gradient computations for specific functions. This is particularly useful for handling non-differentiable functions and avoiding issues such as vanishing gradients or dying ReLU problems. By customizing the gradient computation, we can ensure more stable and efficient training of neural networks.

### The `identity` VJP Decorator

```python
from pyneurons.vjp import identity
```

The `identity` VJP decorator is used to make the gradient of the function it wraps equal to 1, regardless of the input. This can be useful for functions where we want to bypass the standard gradient computation and ensure that the gradient is propagated without any modification.

```python
from jax import custom_vjp

def identity(function):
    wrapper = custom_vjp(function)

    def forward(x):
        return function(x), None

    def backward(_, gradient):
        return (gradient,)

    wrapper.defvjp(forward, backward)
    return wrapper
```

### The `sign` VJP Decorator

```python
from pyneurons.vjp import sign
```

The `sign` VJP decorator modifies the gradient computation by multiplying the gradient with the sign of the input. This can be useful for functions where the gradient should reflect the sign of the input.

```python
from jax.numpy import sign as sign_function
from jax import custom_vjp

def sign(function):
    wrapper = custom_vjp(function)

    def forward(x):
        return function(x), x

    def backward(x, gradient):
        return (gradient * sign_function(x),)

    wrapper.defvjp(forward, backward)
    return wrapper
```

### Usages

The custom VJP decorators are used in various functions within the `pyneurons` library to ensure stable gradient propagation and to handle non-differentiable functions effectively.

#### The `binary` Function

```python
from pyneurons import binary
```

The `binary` function uses the `identity` VJP decorator to ensure that the gradient is always 1, regardless of the input.

```python
from pyneurons.vjp import identity
from jax.numpy import heaviside

@identity
def binary(x):
    return heaviside(x, 1)
```

#### The `relu` Function

```python
from pyneurons import relu, relun, relu1
```

The `relu` function also uses the `identity` VJP decorator to ensure that the gradient is propagated without modification.

```python
from pyneurons.vjp import identity
from jax.numpy import maximum

@identity
def relu(x):
    return maximum(x, 0)
```

#### The `abs` Function

```python
from pyneurons import abs
```

The `abs` function uses the `sign` VJP decorator to ensure that the gradient is modified by the sign of the input.

```python
from pyneurons.vjp import sign
from jax.numpy import abs as function

abs = sign(function)
```

This is used for the `mae` loss function.

## Loss Function: MAE

The `fit` function in `pyneurons` uses the Mean Absolute Error (MAE) as the default loss function. The MAE is defined as:

```python
from pyneurons import abs
from jax.numpy import mean

def mae(y, yhat):
    return mean(abs(y - yhat))
```

### Why MAE?

- **Stability**: MAE is less sensitive to outliers compared to Mean Squared Error (MSE). This makes it a more stable choice for many applications.
- **Simplicity**: The absolute difference is straightforward to compute and interpret.
- **Gradient Behavior**: The gradients of MAE are more stable and less likely to explode or vanish compared to MSE, especially when dealing with large errors.

## Stability in Neural Networks

Stability in neural networks is crucial for ensuring reliable and efficient training, as well as for producing high-quality models.

The `pyneurons` library leverages several techniques to enhance stability, including the use of vector activation functions, custom vector-Jacobian product (VJP) decorators, and the Mean Absolute Error (MAE) loss function.

The vector activation function combines binary and capped ReLU activations, providing a robust mechanism to handle varying input magnitudes and ensuring that neurons can represent both firing states and rates effectively.

Custom VJP decorators, such as `identity` and `sign`, are employed to define stable gradient computations for non-differentiable functions and preventing issues like vanishing or exploding gradients.

The MAE loss function is preferred for its stability, as it is less sensitive to outliers compared to Mean Squared Error (MSE) and provides more consistent gradient behavior, which is essential for maintaining steady learning rates and avoiding gradient-related problems during training.

Together, these components contribute to a more stable and reliable neural network training process, ultimately leading to the development of high-quality models.
