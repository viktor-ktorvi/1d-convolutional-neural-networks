# Time series classification

A simple model to classify a noisy time series into one of three classes: sin, square or sawtooth wave.

The model consist of two convolutional layers and three fully connected layers and achives an accurracy of around 99% training on a Core i5 CPU for about 15 seconds as it is a relatively simple problem.

<p align="center">
  <img width="640" height="480" src="https://user-images.githubusercontent.com/69254199/124983535-5c85e000-e038-11eb-86e8-cd9254372ac2.png">
</p>

Although the accurracy is high, it's not perfect and it isn't really obvious why it makes the mistakes as can be seen bellow.

![predictions](https://user-images.githubusercontent.com/69254199/124983204-f4cf9500-e037-11eb-98fb-b5792fc8cae9.png)

