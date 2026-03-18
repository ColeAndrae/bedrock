# What is a Tensor?

At their core, tensors are mathematical objects that generalize scalars, vectors, and matrices into higher dimensions. While they are foundational to fields like general relativity, continuum mechanics, and electrodynamics, I first encountered them in their purest form through linear algebra. In the context of deep learning, tensors are indispensable; they allow us to batch massive datasets together, leveraging hardware-level parallelism in a way that a scalar-based autograd engine simply cannot match.

# My First Working Prototype:

Over the past few weeks, I’ve been diving deep into the intersection of linear algebra and neural networks. To my surprise, much of the logic governing the Value class in Andrej Karpathy’s MicroGrad engine can be scaled to incorporate higher-dimensional math, with a few key caveats.In MicroGrad, a Value object acts as a node in a computational graph built for backpropagation. Each object contains two primary member variables: data and grad. These are scalar floats representing the value of the operation and its gradient with respect to the loss function.

For TensorGrad, I needed a more robust data structure capable of handling multidimensional operations. The NumPy array is the perfect fit. Unlike a standard Python list, NumPy arrays are designed for the precise data manipulation required for a tensor engine. Operations like matrix multiplication, transposition, and broadcasting are "baked in," providing the efficiency and functionality needed to make this project feasible.

# Overcoming Obstacles:

My first major hurdle was implementing matrix multiplication. While the forward pass of a dot product between two tensors is straightforward, making it differentiable was less intuitive. After some research, I discovered that differentiating matrix operations is remarkably similar to scalar multiplication. In a traditional scalar multiplication, the backward pass involves multiplying the incoming gradient by the value of the other operand. By applying the Jacobian, a matrix containing all first-order partial derivatives of a vector-valued function, the equations for the backward pass remain nearly identical, provided the adjacent tensors are properly transposed to align their dimensions.

Scalar Backward Pass:

def backward():
    self.grad += out.grad * other.data
    other.grad += out.grad * self.data
    
Tensor (Matrix) Backward Pass:

def backward():
    self.grad += out.grad @ other.data.T
    other.grad += self.data.T @ out.grad
    
# What are my Next Steps?

Transitioning to a tensor-based architecture moves the engine from a mathematical curiosity to a functional tool. By replacing nested loops with vectorized operations, TensorGrad significantly reduces overhead. This foundation is the first step toward building more complex architectures, such as Convolutions or Transformers, where scalar computation would simply be too slow to survive.
