# Time series classification

A simple model to classify a noisy time series into one of three classes: sin, square or sawtooth wave.

The model consist of two convolutional layers and three fully connected layers and achives an accurracy of around 99% training on a Core i5 CPU for about 15 seconds as it is a relatively simple problem.

Image 1. The confusion matrix

![cm](https://user-images.githubusercontent.com/69254199/124400783-63091480-dd25-11eb-8c6b-c1e108175544.png)

Below are a few examples of the inputs.

Image 2. A sine wave

![sin](https://user-images.githubusercontent.com/69254199/124400725-0279d780-dd25-11eb-891a-71b7f1d9ea93.png)

Image 3. A square wave

![noisy sqaure wave](https://user-images.githubusercontent.com/69254199/124400729-04dc3180-dd25-11eb-9917-c4c238d305dd.png)

Image 4. A sawtooth wave

![saw](https://user-images.githubusercontent.com/69254199/124400731-06a5f500-dd25-11eb-862a-a15f6ea19ce2.png)

