# NeuralArt-Website 🎨🖌
NeuralArt a website implementing  Neural Style Transfer

![Author](https://img.shields.io/badge/author-vnaazleen-blue)
[![HitCount](http://hits.dwyl.com/vnaazleen/NeuralArt-Website.svg)](http://hits.dwyl.com/vnaazleen/NeuralArt-Website)
![Issues](https://img.shields.io/github/issues/vnaazleen/NeuralArt-Website)
![Stars GitHub](https://img.shields.io/github/stars/vnaazleen/NeuralArt-Website)
![Size](https://img.shields.io/github/repo-size/vnaazleen/NeuralArt-Website)


## Neural Style Transfer
Neural style transfer is an optimization technique takes two images 
* A content image .
* A style reference image (such as an artwork by a famous painter) 
and blend them together so the output image looks like the content image, but  “painted” in the style of the style reference image.
* Gatys et al(A Neural Algorithm of Artistic Style) introduced a way to use Convolutional Neural Network (CNN) to separate and recombine the image content and style of natural images by extracting image representations from response layers in VGG networks.

The below shows how style transfer looks :

![nst](https://user-images.githubusercontent.com/54474853/85428391-c13fdb00-b59a-11ea-9769-01affe0839ec.png)

## Try it
* Try stylizing your images with NeuralArt by some quick steps
* Intall python from [here](https://www.python.org/downloads/)
* Install Flask web framework
```
$ pip install Flask
```
* Install TensorFlow-Hub
```
$ pip install "tensorflow>=1.15,<2.0"
$ pip install --upgrade tensorflow-hub
```
* Clone the repo using
```
$ git clone https://github.com/vnaazleen/NeuralArt-Website.git
```
* Run the server
```
$ python app.py
```
* You are ready to stylize you images now 

## Demo
<img src="https://user-images.githubusercontent.com/54474853/94331428-d9842380-ffe9-11ea-91d6-ec563c689176.gif" alt="neuralart_website_gif" width="1000" height="600"/> 

## References 
* [Wikipedia NST](https://en.wikipedia.org/wiki/Neural_Style_Transfer#NST)
* [A Neural Algorithm of Artistic Style (Gatys et al.)](https://arxiv.org/pdf/1508.06576.pdf)
* [TensorFlow NST Tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)
* [Keras  NST Tutorial](https://keras.io/examples/generative/neural_style_transfer/)
