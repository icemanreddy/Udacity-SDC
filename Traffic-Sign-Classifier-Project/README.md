## Project: Traffic Sign Classifier
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

# Traffic Sign Classifier


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


# Looking at the  Data

 - The training , test and validation data are all in the form pickle files.
 We read it by using something like below.
 ```
 with open(training_file, mode='rb') as f:
        train = pickle.load(f)
 ```
 - From summarizing the data set we see that the training example counts are highly skewed as in the chart below.
 
![Bar chart](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/barchart.png "Summary")


- The validation set is also similarly skewed.
- I printed a few sample images randomly and noticed that the images aren't centered,the lighting/brightness varies and some are rotated/tilted.
- I also noticed that the color of images is not a  distinguishing feature.No two traffic signs have the same shape but different colors.
- Since some of the traffic signs had less than 300 images.I decided to augment the images and create more training data.

# Data Augmentation
 - I am using the Augmentor library to augment the training images.
 - The drawback is that Augmentor library only works on images that are saved on disk.As part of data augmentation,I write  images from the original pickle file to disk to perform any augmentation tasks.
 - My pipeline is very simple.
     - Rotate the image at any angle from (-15° to 15°) with a probability of 50% of applying this operation.
     - Skew images in a random direction by 0.5 with a probability of 50% of applying this operation.
    ```
     p=Augmentor.Pipeline(temp_path+"/"+str(i),".")
     p.rotate(probability=0.5,max_left_rotation=15,max_right_rotation=15)
     p.skew(0.5,0.5)
     ```
- I then generate 1000 images per traffic sign.
- I then create a pickle file so that I don't have to do this procedure again.

NOTE: The augmented pickle files have multiple dictionarties written to them , therefore we need to repeatedly read it and not just once. I had to do this because I got errors while writing all the 77000 images to the pickle file.

# Data preprocessing
- All images are normalized to between 0 and 1.
- Since the images varied in brightness ,I decided to apply histogram equalisation to the images.
- I convert the images from RGB to YUV,apply the equalisation and then convert them back to RGB.

# Design and test Models

- I first tried with a simple LeNet5 model but wasn't able to cross the 90% accuracy mark on the validation set.
- I tried combinations of original data,augmented data,equalised,unequalised data but LeNet would not give satisfactory results.
- I then decided to add more depth to the network since it looked to be underfitting in case of a large training set.
- I also increased the number of output filters because we have a lot of variations in shapes and simple features.
  The models should be capable of capturing them.
- Since the number of nodes after flattening was very high when compared to LeNet.I added an additional fully connected layer.
- So I now have a model with 3 convolution layers and 3 Maxpooling layers followed by 3 fully connected layers.
- I immediately saw much better results(93.5 %) with this model and decided to continue with this model.
- The choice of convolutional kernels selected is 5x5 in the first layer,3x3 in the second layer and 2x2 in the third layer.
- I chose 5x5 for the initial layer so as to be able to capture major/big features that span across a large part of the image.
- At this point I was getting acceptable validation accuracy but the model was overfitting as the training accuracy was nearly 100% and the validation accuracy used to flatline below <95% or start diverging.
- I started modifying dropout rates to fit the data better.
- I found a dropout rate of 0.2 at all layers to give the optimum results. 
- I can achieve accuracy between 96.5 % and 97.5 % on a consistent basis.
- However there is a always variation on retraining from scratch.This is as big as 1% sometimes.
- I tried changing the filter size to 3x3 all across or to (5x5,3x3,3x3) but the performance was similar to the current model.
- I found 10 epochs to be suffcient to converge for my models.
![My model](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/Figure_1.png "My Model")
![Training accuracy](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/training_accuracy_graph.png "Training accuracy")


- Increasing the batch size kept the validation accuracy nearly the same but the model performed poorly with the examples that I provided.
- To see the results from my other models and the different hyperparameters .Please take a look at Final.ipyb and Testing.md


# Predictions on new images.
- I gathered 9 images from the internet of german traffic signs .
- On repeated running after re-training the same model with the same parameters.
- My model is successfully able to identify 8 out of 9 images nearly always.
- I selected images based on how common/uncommon they are in the training data  and their relative similarities to other images.
![Speed Limit 50 kmph](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/classified_2.png) 
- The 50kmph speed limit is nearly always mis-labled as 80kmph speedlimit.
- I was surprised to see that mis classified image is  the image with the most training data.
- It was also mis classified with high certainity. 
- I think this is because of poor resolution of the input data.
- Any sort of resizing would also negatively affect data such as dirty data ,as there would be some information loss.
- I think the image augmentation images may have played some part as the augmented images are rotated and skewed and are not 32x32 .To get them to be 32x32 again some sort of resizing/cropping would be done which could have affected the '5' in the image to look more like an '8'.
- 6 of the other images were classified were near 100% certaininty.

![Wild animals crossing](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/classified_31.png)
![Children Crossing](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/classified_28.png) 
![Slippery Road](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/classified_23.png)
![Speed Limit 30 kmph](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/classified_1.png)
![End of all speed and passing limits](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/classified_32.png) 
![No entry](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/classified_17.png) 
![Right of way at the next intersection](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/classified_11.png)
![Bumpy Road](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/classified_22.png)
- 'Slippery Road sign" was classsified with 60% certainity.The other signs were 'Beware of ice/snow' ,'Bicycles crosssing' and 'wild animals crossing' in decreasing probability.

- Wild animals crossing was classified with 70% certainty.The  other sign is 'Dangerous curve to the right' 
- But these two are classified correctly from time to time 

# Feature maps 
- Following is the features map after convolution and pooling at Layer1.

### Layer1(24 feature maps side to side)
We can see that in the first layers,the shapes are easier to see .
![Feature Map at layer1](https://github.com/icemanreddy/CarND-Traffic-Sign-Classifier-Project/blob/master/featuremap_layer1.png) 

# Improvements that can be done.
- Selective augmentation of data.
    - Don't augment data in training sets with large data already.
    - Add more operations in the pipeline like salt and pepper noise etc.
- Discard images that are dirty.

# Additional Dependencies
[Augmentor](https://github.com/mdbloice/Augmentor) - It is an image augmentation library in Python for machine learning


[Jupyter Notify](https://github.com/ShopRunner/jupyter-notify) - For browser notifications.

