# Time series denoising autoencoder

A simple [denoising autoencoder](https://en.wikipedia.org/wiki/Autoencoder) with 1d CNNs for denoising sine, square and sawtooth waves of random amplitude, frequency and phase.

The model consists of two convolutional layers and two deconvolutional or transpose convolutional layers and the output of the second convolutional layer represents the bottleneck of the autoencoder.

Bellow are the results of training the network on a Core i5 CPU for about 5 minutes so it's not terribly computationally expensive but, of course, it depends on the total data and the length of the signals. Most results are like the ones below but there were also some examples that failed and were even noisier.

![denoised signals](https://user-images.githubusercontent.com/69254199/124969311-d44b0f00-e026-11eb-9557-22653a5871c0.png)

The results are pretty good and compared to the anticausal averaging filter bellow it's a lot better at handling high frequency parts of the signal. Perhaps a bilateral filter would have been a lot closer to what the network is doing.

![filtered](https://user-images.githubusercontent.com/69254199/124969323-d745ff80-e026-11eb-93d1-39577fb436f2.png)

