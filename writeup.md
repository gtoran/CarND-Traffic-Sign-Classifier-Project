# **Traffic Sign Recognition** 

## Final Project Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/gtoran/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set by running the ```len()``` function on ```n_train``` & ```n_test``` to determine training and test sizes, and also to determine the amount of unique classes. Finally, by accessing the first element in the dataset, it's possible to determine the image shape by examining the ```shape``` attribute.

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is [32, 32, 3] *(the last attribute indicates that we are dealing with a 3-channel image)*
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook. I was able to plot the signs in a grid using ```matplotlib``` and looping through all unique training labels, selecting any image associated to said label and appending to an array.

Afterwards, I determine grid rows based on the amount of unique labels divided by the desired amount of columns (that way I can easily change datasets without refactoring hard-coded values).

![Exploratory Visualization](https://gtoran.github.io/repository-assets/CarND-Traffic-Sign-Classifier-Project/writeup-dataset-1.png)

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth, sixth and seventh code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale based on the research carried out by Yann LeCun. His testing determined that shape precedes color in order to determine if a traffic sign is present, and potentially filter what type of sign we are considering. Grayscale is applied to train, test and valid datasets.

Examination of the dataset is easy accomplished by mimicking the same process used above:

![Exploratory Visualization](https://gtoran.github.io/repository-assets/CarND-Traffic-Sign-Classifier-Project/writeup-dataset-2.png)

After converting images to grayscale, I normalized the train and test datasets. My research took me to a very interesting statement on [StackExchange](http://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn/185857#185857) which explains how it is possible to end up continuously over-compensating each feature, thus decreasing accuracy. Mean values for the datasets show a difference before and after normalization.

![Exploratory Visualization](https://gtoran.github.io/repository-assets/CarND-Traffic-Sign-Classifier-Project/writeup-dataset-3.png)

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

My current version of the code doesn't augment data, so I use the originally provided validation data. The dataset expansion, shuffle and split would take place in code cells 10 and 11. I expect data augmentation would improve accuracy even further, although in this scenario I would need to chop the dataset on my own to separate a validation set.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the twelfth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 30x30x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 30x30x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x32 				    |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 12x12x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32    				|
| Flatten   	      	| 2x2 stride,  output 800 units 				|
| Fully connected		| output 600 units								|
| RELU					|												|
| Dropout   	      	|  				                                |
| Fully connected		| output 400 units  						    |
| RELU					|												|
| Dropout   	      	|                                				|
| Fully connected		| output 43 units.  							|
|						|												|

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the thirteenth cell of the ipython notebook. 

To train the model, I used 12 epochs and a batch size of 128. My accuracy levels peak around the 5th epoch and level off at the 11th. Both batch size and learning rate parameters were left intact.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the fourteenth cell of the Ipython notebook. It's based entirely on the LeNet lab solution.

My final model results were, without augmenting data:
* validation set accuracy of max. 0.97, final 0.93
* test set accuracy of 0.964

*(**NOTE**: I accidentally set a 0.5 keep_prob on the evaluate method, which was weighing down my results by at least 5 points).*
*(**NOTE 2**: Even though I used the udacity-carnd AWS instance, a driver/API mismatch caused the GPU to not kick in. I'd like to fix this in order to have faster training times in the future).*

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Traffic Sign](https://gtoran.github.io/repository-assets/CarND-Traffic-Sign-Classifier-Project/other-traffic-signs/sign-1.png)

![Traffic Sign](https://gtoran.github.io/repository-assets/CarND-Traffic-Sign-Classifier-Project/other-traffic-signs/sign-2.png)

![Traffic Sign](https://gtoran.github.io/repository-assets/CarND-Traffic-Sign-Classifier-Project/other-traffic-signs/sign-3.png)

![Traffic Sign](https://gtoran.github.io/repository-assets/CarND-Traffic-Sign-Classifier-Project/other-traffic-signs/sign-4.png)

![Traffic Sign](https://gtoran.github.io/repository-assets/CarND-Traffic-Sign-Classifier-Project/other-traffic-signs/sign-5.png)

My initial doubts concerned the general caution sign: the distortion and small size make me think it will throw it off and consider it something else, like the pedestrian crossing sign.

![Traffic Sign](https://gtoran.github.io/repository-assets/CarND-Traffic-Sign-Classifier-Project/pedestrian.png)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the seventeenth cell of the Ipython notebook.

Here is the grid of the new traffic signs:

![Traffic Sign Grid](https://gtoran.github.io/repository-assets/CarND-Traffic-Sign-Classifier-Project/writeup-plot-german-signs.png)

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit      		| Speed limit  									| 
| General caution		| General caution								|
| Keep right    		| Keep right									|
| Speed limit	   		| Speed limit					 				|
| Priority road 		| Priority road      							|

The model was able to correctly guess all traffic signs, providing complete accuracy. On a larger dataset I would expect to see some errors given the error rate.

Compared to the test dataset, accuracy with these five images is higher (100%) than what we saw in the previous model (%96). Performance on new data is better than the training dataset, so fortunately it doesn't look like the model is overfitting (as in it's not "memorizing" training data instead of generalizing from trend).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the eighteenth cell of the Ipython notebook. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit       							| 
| 1.00     				| General caution								|
| 1.00					| Keep right									|
| .99	      			| Speed limit					 				|
| 1.00				    | Priority road      							|

In all cases, the algorithm is very accurate, contrary to my initial thoughts about the general caution sign.

### Areas of improvement

I would focus further development on improving the accuracy of the project by augmentating data. My initial thoughts on how to proceed have been laid out, but not implemented. To achieve this, I would create and append randomly up/downscaled, rotated and/or shifted versions of the images in the dataset. To ensure consistency, I would deprecate the validation dataset read that is carried out in the first code cell, and I would obtain a new one by shuffling and spliting the augmentated dataset. I'm estimate these changes to bring a 1-2% increase in accuracy, and presumably improve probability on the first sign mentioned above.
