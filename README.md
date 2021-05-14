# Traffic Sign Recognition
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project is, to train a deep neural network in order to classify traffic signs. The deep neural network is based on the [LeNet](https://en.wikipedia.org/wiki/LeNet#:~:text=LeNet%20is%20a%20convolutional%20neural,a%20simple%20convolutional%20neural%20network.) architecture and trained on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

## Project Organization
I've based the folder structure on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) structure. Its a logical structure and therfore easy to collaborate.

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    ││
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py

## Project goals

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Installation & running

I've had several issues getting the [CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) up and running locally. The Udacity workspace contains tensorflow 1.3 so it's almost obsolete since the current version is 2.5. I'm quite experienced with tensorflow 2.x so therefore i've decided to take a more modern approach.

1. Download the [dataset)[https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip]
2. unzip it and put it into the `data/processed` folder
3. Create a new virtual environment `mkvirtualenv traffic_sign_recognition` and activate it `workon traffic_sign_recognition`
4. Now install the requirements `pip install -r requirements.txt`
5. Everything should be present, now run `jupyter lab` and open the notebook in the notebooks folder

## Project Development

### Data Set Summary & Exploration

I've read the pickle (train/valid/test) files and split them into features (X) and labels (y). I've also read the `signames.csv` into a pandas dataframe so i can use it to create the variable `class_names` which i'll be using to map the `class_id` to the actual label. I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

Here is an exploratory visualization of the data set. It is a bar chart showing how all the sign names with their number of images in the trainingset. I've sorted the sign names based on the number of images so we can easily identify which sign names are more and less freqent. I've a rule of thumb that states _"preferably >= 1000 training samples per class"_. Therefore i've added an threshold which highlight the abundant classes and the classes that are likely to perform less.

[image1]: ./reports/figures/training_images_class_distribution.png "Class distribution"
![alt text][image1]

### Design and Test a Model Architecture

#### Data Preprocessing
As you can see there is a great imbalance between the number of training images per class. So in order to take this into account, i've calculated the weight per class `class_weight`. This variable will be used in the training phase to balance the network by applying the class weights in the network.

Since i'm using Tensorflow 2.x i might aswell use the tensorflow dataset to manage the data.
```python
# create tensorflow datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
```

Next i've preprocessed the datasets. The trainingset is shuffled so the order of the training images doesn't affect the network to much. Next i've applied augmentation on the trainingset, so the network will become more robust. As you can see in `src/features/build_features` i've randomized the (hue, saturation, brightness, contrast) of the training images. All datasets were preprocessed by the same actions (standardization & grayscaling).
```python
# preprocess the datasets + apply augmentation
train_ds = build_features.preprocess(train_ds, batch_size=batch_size, shuffle=True, augment=True)
val_ds = build_features.preprocess(val_ds, batch_size=batch_size)
test_ds = build_features.preprocess(test_ds, batch_size=batch_size)
```

Personally i think it is a good practise to visualise a sample of the trainingset, se we can _"see"_ what we're working with. Here is an example of the original training images.

[image2]: ./reports/figures/training_examples.png "Original training images"
![alt text][image2]

And are some augmented and preprocessed training images.

[image3]: ./reports/figures/training_examples_preprocessed.png "Preprocessed training images"
![alt text][image3]

#### Model architecture

As mentioned before, iv've based the network architecture on the [LeNet](https://en.wikipedia.org/wiki/LeNet#:~:text=LeNet%20is%20a%20convolutional%20neural,a%20simple%20convolutional%20neural%20network.) architecture. `src/models/train_model`. I've adjusted the imput layers so it matches the image dimensions `(32,32,1)` after preprocessing, and the ouput layer to match the number of classes `n_classes`. I've also added some dropout layers, so the model will become more robust.

[image4]: ./reports/figures/model_summary.png "Model summary"
![alt text][image4]

#### Model training

Now that the data is prepared and the model architecture is defined, i can train the model. In `src/models/train_model` is the training code. As you can see i've:
- compiled the model with the Adam optimizer, and started with an initial learing rate of (0.001)
- the model trains for a maximum of 50 epochs
- added callbacks 
 - EarlyStopping: prevents overfitting by monitoring the loss. If the validation loss doesn't decrease any futher the function will stop the epochs.
 - ReduceLROnPlateau: monitors also the validation loss. If the validation loss doesn't decrease any futher the function decrease the learning rate. So we'll find the global optimum.
- added the class weights, to counter the class imbalance
- because i'm using tensorflow datasets the data is allready batched in sizes of 128.

As you can see in the training history:

[image5]: ./reports/figures/training_history.png "training history"
![alt text][image5]

The accuracy of the training and the validation, steadily grows above the `threshold = 0.93`, while the loss is decreasing. The ReduceLROnPlateau occasionally drops the learning rate and the EarlyStopping stops the training before the 50th epoch.

I've evaluated the model on all (training, validation, test) datasets.

[image6]: ./reports/figures/model_accuracy_on_datasets.png "Model accuracy"
![alt text][image6]

My final model results were:
- training set accuracy of 0.996
- validation set accuracy of 0.933
- test set accuracy of 0.909




 




#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 




































## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

