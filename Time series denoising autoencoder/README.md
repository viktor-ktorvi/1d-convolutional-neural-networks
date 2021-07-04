# Time series denoising autoencoder

A simple [denoising autoencoder](https://en.wikipedia.org/wiki/Autoencoder) with 1d CNNs for denoising sine, square and sawtooth waves of random amplitude, frequency and phase.

The model consists of two convolutional layers and two deconvolutional or transpose convolutional layers and the output of the second convolutional layer represents the bottleneck of the autoencoder.

Below are a couple of results from training a network for each waveform on a Core i5 CPU for about 5 minutes so it's not terribly computationally expensive but, of course, it depends on the total data and the length of the signals. Most results are like the ones below but there were also some examples that failed and were even more noisier.

Image 1. A sine wave

![sine denoised](https://user-images.githubusercontent.com/69254199/124363308-c1a59400-dc3a-11eb-9a63-8cd4ae643209.png)

Image 2. A square wave

![square wave](https://user-images.githubusercontent.com/69254199/124397790-49120680-dd12-11eb-9d67-621a5ca670eb.png)

Image 3. A sawtooth wave

![saw wave](https://user-images.githubusercontent.com/69254199/124397798-58914f80-dd12-11eb-8fac-8c0b01f24550.png)


## TODO Compare with anticausal filter
