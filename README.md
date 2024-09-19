# CNN-based Age Estimation

This project demonstrates how to estimate the age of a person from an image using deep learning techniques. The model is based on the ResNet-50 architecture, pre-trained on the ImageNet dataset, and fine-tuned for age estimation as a regression task.

<p align="center">
  <img src="https://github.com/masoud-ml/CNN-based_Age_Estimation/blob/main/misc/image.png" style="width:550px; height:400px">
</p>

## Overview

Age estimation is the task of predicting a person's age based on facial images. This project leverages pre-trained models, particularly ResNet-50, along with facial recognition techniques to perform this task efficiently.

## Dataset

This project is based on the UTKFace dataset, a large-scale dataset containing over 20,000 facial images with corresponding age, gender, and ethnicity labels. The dataset covers a wide range of ages (from 0 to 116 years) and is highly diverse in terms of ethnicity and gender.

## Model Architecture

The core of this project is based on the **ResNet-50** model architecture, which is a deep convolutional neural network with 50 layers, pre-trained on the ImageNet dataset.

1. **ResNet-50**:
   - The model used is **ResNet-50**, a variant of the ResNet (Residual Network) family that is 50 layers deep.
   - ResNet-50 is designed with **residual connections**, allowing the network to skip layers during training. This helps combat the **vanishing gradient problem** and enables deeper networks to learn more effectively.

2. **Pre-trained Weights**:
   - The model is initialized with pre-trained weights from the ImageNet dataset (`ResNet50_Weights.IMAGENET1K_V2`). This transfer learning approach allows the model to leverage knowledge from the general image classification task (over 1,000 classes) and apply it to age estimation, improving performance and reducing training time.

3. **Modifications for Age Estimation**:
   - Since age estimation is a **regression task**, the final fully connected layer of the ResNet-50 is replaced with a single neuron that outputs a continuous value (the predicted age). 
   - In the original ResNet-50, the final layer outputs probabilities across multiple classes (for classification tasks). Here, we modify it to predict a single numeric value representing age.

This architecture allows the model to efficiently learn facial features and estimate a person’s age from an image.
