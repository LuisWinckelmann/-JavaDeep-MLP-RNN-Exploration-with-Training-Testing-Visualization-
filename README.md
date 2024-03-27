<!-- README.md -->
<!-- Project Top -->
<a name="readme-top"></a>


<h1 align="center">JavaDeep: MLP & RNN implementation from scratch in Java</h1>

  <p align="center">
    A project exploring Multilayer Perceptrons (MLP) and Recurrent Neural Networks (RNN) with implementations from scratch in Java.
<!---    <br/>
    <a href="https://github.com/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java/blob/main/gfx/MLPGeometry.gif">Demo MLP</a>
   &
   <a href="https://github.com/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java/blob/main/gfx/RNNTrajectory.gif">Demo RNN</a> 
-->
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#classes-and-functionality">Classes and Functionality</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

JavaDeep is a project dedicated to exploring Multilayer Perceptrons (MLP) and Recurrent Neural Networks (RNN) core 
functionalities through a pure Java implementation. To gain a profound understanding of their inner workings, I 
implemented the forward pass, backward pass (through time), and stochastic gradient descent with a momentum term for 
both types of models. To test the functionality of both the MLP & RNN, <a href="#usage">three toy problems</a> are 
provided.


| MLP predictions each 5 epochs during training | Fully trained RNN after 25k training epochs |
|:-------------------------:|:-------------------------:|
|[![Example Visualization of the MLP][product-screenshot]](gfx/MLPGeometry_small.gif) | [![Example Visualization of the RNN][product-screenshot2]](gfx/RNNTrajectory_small.gif) |

Corresponding to:

| Binary Classification    | Past-dependent spiral movement|
|:-------------------------|:-------------------------|
|The problem simulated is a non-linear classification problem. Over time the MLP learns to separate the background (Class 1, black) from the 2 visible circles (Class 2, white). The model predictions for Class 2 are highlighted in blue every 5 epochs. | The problem simulated is a past-dependent movement in form of a spiral. The fully trained RNN is able to quickly close the gap to the spiral and predict it's future path based on previous movement. Orange corresponds to the predictions and blue corresponds to the ground truth spiral.|


## Getting Started

### Prerequisites

All you need to run this project is Java. This project was originally implemented with **SDK 15.0.2.**, but any [newer version](https://www.oracle.com/java/technologies/downloads/) should also work.

### Installation

Clone the repo
   ```sh
   git clone https://github.com/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java.git
   ```

## Classes and Functionality
- Networks:
  - `MultiLayerPerceptron`: Basis for creating MLP of any size with sigmoid activation function. Includes forward pass, backward pass and stochastic gradient decent (SGD) with momentum,
  - `RecurrentNeuralNetwork`: Basis for creating any RNN with tanh activation function. Includes forward pass, backpropagation through time and SGD with momentum.
- Toy problems:
  - `MLPXOR.java`: Classic XOR toy problem with training and testing.
  - `MLPGeometry.java`: Non-linear binary classification problem (see <a href="#about-the-project">here</a>) with training, testing and visualizations.
  - `RNNTrajectory.java`: Past-dependent toy problem of simulating a spiral movement (see <a href="#about-the-project">here</a>) with training, testing and visualizations.

## Roadmap

- [x] Implement MLP
- [x] Implement RNN
- [x] Create Toy Test Datasets
- [x] Include Visualizations
- [x] Clean-Up Code
- [x] Move project from private to public
- [x] Finalize README & Visualizations
- [ ] Test if cloning and executing the repository from scratch works
  

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.


## Contact
[![LinkedIn][linkedin-shield]][linkedin-url] <br>
Luis Winckelmann  - luis.winckelmann@gmx.com <br>
Project Link: [https://github.com/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java](https://github.com/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java.svg?style=for-the-badge
[license-url]: https://github.com/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/luiswinckelmann
[product-screenshot]: gfx/MLPGeometry_small.gif
[product-screenshot2]: gfx/RNNTrajectory_small.gif

