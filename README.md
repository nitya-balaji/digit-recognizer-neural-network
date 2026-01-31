# Digit Recognizer Neural Network

A from-scratch implementation of a neural network for handwritten digit recognition using the MNIST dataset. This project achieves **~85% accuracy** on the validation set.

## Overview

This project implements a neural network from the ground up, going underneath the abstraction layers of TensorFlow and PyTorch to understand the fundamental mathematics and mechanisms that power deep learning. Built using only NumPy, Pandas, and linear algebra, this implementation demonstrates the core concepts of neural networks without relying on high-level frameworks.

## Key Concepts Implemented

This neural network was built using the following fundamental techniques:

- **Forward Propagation** - Computing activations layer by layer through the network
- **Backpropagation** - Computing gradients by propagating errors backward through the network
- **Gradient Descent** - Optimizing weights and biases to minimize the loss function
- **Softmax Activation** - Converting output layer values into probability distributions
- **One-Hot Encoding** - Representing categorical labels as binary vectors

## Tools & Libraries

- **NumPy** - Matrix operations and numerical computations
- **Pandas** - Data manipulation and loading the MNIST dataset
- **Matplotlib** - Visualization of predictions and training progress

## Dataset

The model is trained on the **MNIST dataset**, a collection of 70,000 handwritten digits (0-9) commonly used as a benchmark for image classification tasks. Each image is 28x28 pixels in grayscale.

## Results

The neural network achieves approximately **85.20% accuracy** on the validation set, demonstrating that even a simple from-scratch implementation can achieve reasonable performance on this classic machine learning problem.

## Math and Rough Work

For a detailed breakdown of the calculus behind backpropagation, along with all the relevant derivations, notes, and rough work that went into building this project, you can find my notes here: [Backpropagation Calculus + Digit Recognizer Neural Network - Math & Notes](https://drive.google.com/file/d/16RVa49mzZHu2DrikHk-R9L2lOwAC_yRH/view?usp=sharing)

## Inspiration

This project was inspired by [3Blue1Brown's neural network series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), which provides excellent visual explanations of how neural networks learn through gradient descent and backpropagation. 


