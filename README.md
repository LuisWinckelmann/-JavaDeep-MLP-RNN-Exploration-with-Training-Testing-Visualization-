<!-- README.md -->
<!-- Project Top -->
<a name="readme-top"></a>


<h1 align="center">JavaDeep: MLP & RNN implementation from scratch in Java</h1>

  <p align="center">
    A project exploring Multilayer Perceptrons (MLP) and Recurrent Neural Networks (RNN) with implementations from scratch in Java.
    <br/>
    <a href="https://github.com/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java/blob/main/gfx/MLPGeometry.gif">View Demo</a>
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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

JavaDeep is a project dedicated to exploring Multilayer Perceptrons (MLP) and Recurrent Neural Networks (RNN) through 
pure Java implementations. To gain a profound understanding of their inner workings, I implemented the forward pass, 
backward pass, and stochastic gradient descent with a momentum term for both types of models. To test the functionality,
<a href="#usage">three different classes</a>, along with toy datasets and visualizations, are provided.

[![Example Visualization of the MLP][product-screenshot]](gfx/MLPGeometry_small.gif)

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

All you need to run this project is Java. A [newer version](https://www.oracle.com/java/technologies/downloads/) should work. This project was originally implemented with **SDK 15.0.2.**

### Installation

Clone the repo
   ```sh
   git clone https://github.com/LuisWinckelmann/JavaDeep-MLP-RNN-from-scratch-in-Java.git
   ```

<!-- USAGE EXAMPLES -->
## Usage

- `MLPXOR.java`: Training & Testing of the MLP on a toy XOR problem
- `MLPGeometry.java`: Training & <a href="#about-the-project">Visualizations</a>  of the MLP on a toy classification problem that requires a non-linear solution.
- `RNNTrajectory.java`: Training & Testing of the RNN on a toy dataset of a spiral sequence that the RNN needs to learn.

<!-- ROADMAP -->
## Roadmap

- [x] Move project from private to public
- [x] Finalize README
    - [x] Description
    - [x] Installation/Setup
    - [x] Example Usage & Create/Render Example Gif
- [ ] Test if cloning and executing the repository from scratch works
  
<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- CONTACT -->
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

