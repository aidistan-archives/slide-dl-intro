dl-intro
========
A shallow introdcution to deep learning.

## Prelude

### Artificial neuron

- The basic building blocks
- A mathematical model to mimic biological neurons
- Map multiple inputs to one output

### Activition function 

- Sigmoid function is a common choice
- Other choices include 
	- tanh function \\(f(z) = \frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}\\)
    - rectified linear function \\(f(z) = \max(0,x)\\)

### Neural network

- Hook many neurons together to get a neural network
- So defined input layer, hidden layer, output layer
- bias unit, input unit, hidden unit, output unit

### Forward propagation

- With these equations,  we can take advantage of fast linear algebra routines to perform calculations easily and quickly, if only we have the parameters.

### Training NNs

- Minimize the cost function / loss function
- A regularization term (also called a weight decay term) that tends to decrease the magnitude of the weights, and helps prevent overfitting
	- Weight decay parameter \\(\lambda\\) controls the relative importance of the two terms
- It's a non-convex function, gradient descent is susceptible to local optima
	- However, in practice gradient descent usually works fairly well

### Gradient descent

- It's hard to determine \\(\alpha\\), a convex optimization problem
- We can easily use any convex optimization algorithm (like LBGFS, which is much better than gradient descent algorithm) to update the parameters, if only we have all those partial derivatives.
- How to compute them efficiently?

### Backpropagation

- It's important to initialize the parameters randomly, rather than to all 0’s. If all the parameters start off at identical values, then all the hidden layer units will end up learning the same function of the input.

### NN seems promising

- No need for hand-crafted features

### Towards deeper NNs ?

- It's natural to increase the number of hidden layer. But why?

## Why go deep ?

### Visual cortex is hierarchical

- Cat's visual cortex (Hubel, 1962)
- Human visual system

### Different Levels of Abstraction

- Natural progression from low level to high level structure as seen in natural complexity
– Easier to monitor what is being learnt and to guide the machine to better subspaces
– A good lower level representation can be used for many distinct tasks

### Efficient representation

## Rolling in the deep 

Rolling in the deep (在黑暗中翻滚)

### Deep neural networks

- Related to the multilayer perceptron
	- A multilayer perceptron (MLP) is a feedforward artificial neural network model that maps sets of input data onto a set of appropriate outputs. A MLP consists of multiple layers of nodes in a directed graph, with each layer fully connected to the next one. Except for the input nodes, each node is a neuron (or processing element) with a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training the network. MLP is a modification of the standard linear perceptron and can distinguish data that are not linearly separable.
	- The multilayer perceptron consists of three or more layers (an input and an output layer with one or more hidden layers) of nonlinearly-activating nodes and is thus considered a deep neural network.
- We'll talk about Softmax regression later

### Difficulty of training DNNs

- The brain can learn from unlabeled data
- If the last few layers in a neural network have a large enough number of neurons, it may be possible for them to model the labeled data alone without the help of the earlier layers. Hence, training the entire network at once with all the layers randomly initialized ends up giving similar performance to training a shallow network (the last few layers) on corrupted input (the result of the processing done by the earlier layers)

### Other models dominanted the world

- People turned to study "machine learning", which seems more insteresing
- Enemy #1 SVM and other algorithms, like boosting, developed fast in this period

### Hand-crafted visual features

- State-of-the-art visual features
- Until 2006

> 一位痴心的老先生Hinton，他坚持了下来，并最终（和其它人一起）提出了一个实际可行的深度学习框架。

- On the other hand, big data era was emerging. We need more powerful algorithms.

## Stacked autoencoder

### Autoencoder

- Suppose we have only a set of unlabeled training examples
- An autoencoder neural network is an unsupervised learning algorithm that applies backpropagation, setting the target values to be equal to the inputs
- The aim of an auto-encoder is to learn a compressed, distributed representation (encoding) for a set of data, typically for the purpose of dimensionality reduction.
- When the hidden layers are larger than the input layer, an autoencoder can potentially learn the identity function and become useless; however, experimental results have shown that such autoencoders might still learn useful features in this case.

### Sparse AE

- Even when the number of hidden units is large (perhaps even greater than the number of input pixels), we can still discover interesting structure, if we impose a sparsity constraint on the hidden units

### Stacked AE

- Multiple sparse autoencoder + Softmax regression
- Has the same problem existing in training DNNs

### Denoising AE

- Steps
	- Corrupt the input (e.g. set 25% of inputs to 0)
	- Reconstruct the uncorrupted input
	- Use uncorrupted encoding as input to next level
- More robust model

## Time to have an overview

## DBM & DBN

### RBM

- Bipartite graph, one visible layer and one hidden layer. "Restricted" indicates there is no edge within one layer.
- Given all visible variables <i>v</i>, all hidden variables are mutual independence. We could get the distribution of all <i>h</i> based on p(h|v), and get back <i>v</i> in turn based on p(v|h).
- Thus, <i>h</i> are just another representation of <i>v</i>

### CNN

- First successfully trained deep structure
- Before greedy layer-wise training

## Industry

### Microsoft

- Proposed the first successful deep learning models for speech recognition in 2009
- Made a new breakthrough in speech translation technology in 2012
- Plans to make skype translator alive within 2014

### Google

- Initially established by Andrew Ng, however he moved to lead AI group at Baidu
- Led by Jeff Dean and Geoffrey Hinton by now

## Challenges of Deep Learning

### Understanding

- Better theoretical understanding
- Convex learning is invariant to the order in which sample are presented
- Human learning isn't like that: we learn simple concepts before complex ones. The order in which we learn things matter.

## Scaling

### Model parallelism

An example of model parallelism in DistBelief. A five layer deep neural network with local connectivity is shown here, partitioned across four machines (blue rectangles). Only those nodes with edges that cross partition boundaries (thick lines) will need to have their state transmitted between machines. Even in cases where a node has multiple edges crossing a partition boundary, its state is only sent to the machine on the other side of that boundary once. Within each partition, computation for individual nodes will the parallelized across all available CPU cores.

### Data parallelism

Left: Downpour SGD. Model replicas asynchronously fetch parameters w and push gradients w to the parameter server. Right: Sandblaster L-BFGS. A single ‘coordinator’ sends small messages to replicas and the parameter server to orchestrate batch optimization.
