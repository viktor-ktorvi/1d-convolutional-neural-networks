# Time series denoising autoencoder

Training a simple [denoising autoencoder](https://en.wikipedia.org/wiki/Autoencoder) with 1d CNNs.
The training data consisted of sine, square, sawtooth and sinc functions with additive white gaussian noise.

<p align="center">
  <img src="images/training_data.png"/>
</p>

The network, consisting of a few convolutional and the same number of transpose convolutional layers,
learns a non-linear filtering function and performs noticeably better than a simple anti-causal average filter.

<p align="center">
  <img src="images/evaluation_results.png"/>
</p>


<p align="center">
  <img src="images/anticausal_average.png"/>
</p>