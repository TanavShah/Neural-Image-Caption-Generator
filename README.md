# Neural Image Caption Generator

Implementation of Neural Image Caption Generator, which takes as input an image and outputs a description of the image as text. An attention based model, 
trained on MS-COCO dataset, uses Inception V3 for preprocessing and encoder-decoder model for training.

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p2/download.png" width="300" height="300" />
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p2/download%20(2).png" width="500" height="300"/>
</p>

<p align="center">
Generated Caption : young child rides a large snow board a very snowy slope.
</p>

Automatically generating captions of an image is a task very close to the heart of scene understanding — one of the primary goals of computer vision.
Not only must caption generation models be powerful enough to solve the computer vision challenges of determining which objects are in an image, but they mut also be capable of capturing and expressing their relationships in a natural language.
One of the most curious facets of the human visual system is the presence of attention.
As opposed to pack a whole picture into a static portrayal, attention allows for salient features to dynamically come to the forefront as needed. 
This is particularly significant when there is a ton of clutter in a picture.
Using representations (such as those from the top layer of a convnet) that distill information in image down to the most salient objects is one powerful arrangement that has been broadly embraced in past work.
Unfortunately, this has one potential drawback of losing information which could be useful for richer, more descriptive captions. Using more low-level representation can help preserve this information.
However working with these features necessitates a powerful mechanism to steer the model to information important to the task at hand.


## The Dataset

The dataset used here is the Microsoft COCO: Common Objects in Context dataset. COCO is a large-scale object detection, segmentation, and captioning dataset. 
It contains images of complex everyday scenes containing common objects in their natural context. 
Objects are labeled using per-instance segmentations to aid in precise object localization.
The dataset contains photos of 91 objects types that would be easily recognizable by a 4 year old.
The dataset contains over 82,000 images, each of which has at least 5 different caption annotations.
We use a subset of 30,000 captions and their corresponding images to train our model. Choosing to use more data would result in improved captioning quality.

Here is an example from the dataset.

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/image.png" width="400" height="400" />
</p>

<p align="center">
Caption : <start> a woman in a blue dress is playing tennis <end>
</p>

## Pre-Processing

### 1. We use InceptionV3 (which is pretrained on Imagenet) to classify each image. We extract features from the last convolutional layer.

First, we convert the images into InceptionV3's expected format by:

* Resizing the image to 299px by 299px
* Preprocess the images using the preprocess_input method to normalize the image so that it contains pixels in the range of -1 to 1, which matches the format of the images used to train InceptionV3.

Next we create a tf.keras model where the output layer is the last convolutional layer in the InceptionV3 architecture. 
The shape of the output of this layer is 8x8x2048. 
We use the last convolutional layer because we are using attention in this example.

* Forward each image through the network and store the resulting vector in a dictionary (image_name --> feature_vector).
* After all the images are passed through the network, pickle the dictionary and save it to disk.

Pre-process each image with InceptionV3 and cache the output to disk.

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/InceptionV3_model.png" />
</p>

<p align="center">
The InceptionV3 model architecture
</p>

### 2. Preprocessing and tokenizing the captions.

* First, tokenize the captions (for example, by splitting on spaces). This gives us a vocabulary of all of the unique words in the data (for example, "surfing", "football", and so on).
* Next, limit the vocabulary size to the top 5,000 words (to save memory). Replace all other words with the token "UNK" (unknown).
* Then create word-to-index and index-to-word mappings.
* Finally, pad all sequences to be the same length as the longest one.


## The Model

The model design is propelled by the [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) paper.

* The model extracts the features from the lower convolutional layer of InceptionV3 giving us a vector of shape (8, 8, 2048).
* It then squashes that to a shape of (64, 2048).
* This vector is then passed through the CNN Encoder (which consists of a single Fully connected layer).
* The RNN (here GRU) attends over the image to predict the next word.

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/Model_Architecture.png" />
</p>


## Training Process

* Extract the features stored in the respective .npy files and then pass those features through the encoder.
* The encoder output, hidden state (initialized to 0) and the decoder input (which is the start token) is passed to the decoder.
* The decoder returns the predictions and the decoder hidden state.
* The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
* Use teacher forcing to decide the next input to the decoder. Teacher forcing is the technique where the target word is passed as the next input to the decoder.
* The final step is to calculate the gradients and apply it to the optimizer and backpropagate.

#### Parameters :

* BATCH_SIZE = 64
* BUFFER_SIZE = 1000
* embedding_dim = 256
* units = 512
* EPOCHS = 20
* Optimizer - Adam
* Loss Function - SparseCategoricalCrossentropy

## Results 

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p1/download%20(1).png" width="200" height="200" />
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p1/download.png" width="400" height="200"/>
</p>

<p align="center">
Predicted Caption :  asian girl and their face stands under an umbrella <end>
</p>

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p6/download.png" width="200" height="200" />
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p6/download%20(2).png" width="400" height="200"/>
</p>

<p align="center">
Predicted Caption :  close up photo taken some vegetables and pickle <end>
</p>

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p10/download.png" width="200" height="200" />
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p10/download%20(2).png" width="400" height="200"/>
</p>

<p align="center">
Predicted Caption :  desk with 2 laptop computer equipment cell phone <end>
</p>

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p7/download.png" width="200" height="200" />
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p7/download%20(1).png" width="400" height="200"/>
</p>

<p align="center">
Predicted Caption :  pile of hotdogs on lunch <end>
</p>

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p4/download.png" width="200" height="200" />
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/p4/download%20(2).png" width="400" height="200"/>
</p>

<p align="center">
Predicted Caption :  <unk> with cut up flower <unk> with a computer and a cup<end> <unk> <end>
</p>



## Observations

The model is trained on Google Colab for 20 Epochs. Each epoch takes approximately 30-50 seconds. 
The model works pretty well and can generate good looking captions.
However, training the model for more number of epochs, does not seem to significantly improve the performance.
On the other hand, increasing the size of dataset from 30,000 captions to 50,000 captions, increases the performance quite well. 
The only problem is that it uses up a lot of space and takes a very long time to train.

The loss plot obtained while training over epochs is shown below :

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/loss.png" />
</p>

There are some drawbacks of the model as well. There is not enough provision for special characters. The generated captions are not always very well formulated. For example, the following caption does not make any sense from the 
given image. This could be improved in future work.

<p align="center">
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/non%20sense/p2/download.png" width="200" height="200" />
  <img src="https://github.com/TanavShah/Neural-Image-Caption-Generator/blob/master/Results/non%20sense/p2/download%20(2).png" width="400" height="200"/>
</p>

<p align="center">
Real Caption :  <start> a large church with a tall brick clock tower in it's center <end>
</p>

<p align="center">
Predicted Caption :  white sheep are shown with a cows in a hillside <end>
</p>

## References and Credits

* [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
* [TensorFlow Image Captioning Tutorial](https://www.tensorflow.org/tutorials/text/image_captioning)



