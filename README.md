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
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   └── __init__.py    <- Makes src a Python module
    │
    ├── data               <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features           <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models             <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization      <- Scripts to create exploratory and results oriented visualizations
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

As you can see in the `notebooks/experiments` folder. i've took an itterative approach in order to increase the model performance. I duplicate the notebook and change ther version number + changelog (in the top) to keep track of all the experiments. I've started with a simple 4 layer custom model, and measured the accuracy during training and on all the datasets. Over time i've added functionality like rescaling, dropout, callbacks in order to improve performance. Once i was satisfied i've migrated the code over to the original notebook. 

During writeup i noticed i've had to achieve an accuracy of 0.93 on the validation set. So i've experimented quite a lot in the original notebook until i've adchieved the threshold. You can probably trace this in the git history of the file. I've:
- changed from a simple 4 layer custom model, a LeNet like architecture
- applied several augmentations (flipping, didn't seem right to me)
- experimented with batch sizes, nr epochs

Eventually the current codebase with the preprocessing, augmentation, and hyperparameters seemed to worked the best.

### Test a Model on New Images

#### Data loading
I've manually searched and downloaded several images from the internet. I've kept a copy of the original in the `data/raw` folder. Next i've manually:
- croped the image to a `square` where the traffic sign was located
- changed the image dimensions to 32,32 pixels
- put the in a folder with the corresponding class_id

Now i can use the preprocessing function to load images directly into a dataset and infer the class_names from the directory names.
```python
### Load the images and plot them here.
custom_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(os.pardir, 'data', 'processed', 'custom_images'),
    shuffle=False,
    image_size=(32, 32),
    batch_size=batch_size
)
```

Now that i have some images i can _"see"_ if they are correct.

[image7]: ./reports/figures/custom_images.png "Custom images"
![alt text][image7]

#### Making predictions

Before i can make any predictions on the data i need to preprocess them on the same steps as i've did on the training data. I don't need to batch them, because in the loading from directory the images where allready batched.

```python
# preprocess the datasets + apply augmentation
custom_ds = build_features.preprocess(custom_ds, batch=False)
```

Now the data is ready, i can make the predictions. Since the output of the model are logits i'll append a softmax layer to the model in order to get values between 0 and 1, which in turn resemble a probability. Now the output of the model (+ softmax) is a probability per class and this is usefull for the predicting. After we've predicted the classes we need to apply and argmax function (per image) to figure out which class has the highest probability. 

| Image (y_true) | Prediction (y_pred) | 
|:---:|:---:| 
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Yield | Yield |
| Yield | Yield |
| Stop | Stop |
| General caution | General caution |
| Speed limit (50km/h) | Speed limit (30km/h) |
| Speed limit (50km/h) | Roundabout mandatory |
| Bumpy road | Bumpy road |
| Road work | Road work |
| Children crossing | Bicycles crossing |
| Speed limit (70km/h) | Speed limit (70km/h) |

#### Analyze performance

The model made the following mistakes.

| Image (y_true) | Prediction (y_pred) | 
|:---:|:---:| 
| Speed limit (50km/h) | Speed limit (30km/h) |
| Speed limit (50km/h) | Roundabout mandatory |
| Children crossing | Bicycles crossing |

The first and the last mistake make sense to me, the signs are quite similar. As for the middle mistake, i cant really tell why.

Now we have the true classes (from the directories) `y_true` and the predicted classes `y_pred`, and we can generate a confusion matrix to inspect which predictions where right and which where wrong. A perfect model would have the same classes predicted as the true values, and the confusion matrix would highlite this in the diagonal line. 

[image8]: ./reports/figures/model_accuracy_confusion_matrix.png "Confusion matrix"
![alt text][image8]

Now let's see how well the custom dataset holds against the other datasets. The performance should be less than the other datasets, because there are far fewer images, and thus a false prediction would have much more impact than the large datasets.

[image9]: ./reports/figures/model_accuracy_on_datasets_and_custom.png "Dataset accuracy comparisson"
![alt text][image9]

#### Top 5 Softmax probabilities

If we take a closer look at the top 5 predictions per image, we'll see that the majority that is correctly classified has a probability of 100%. This implies that the model whas quite confident this is the right classification. I would have expected (hoped) that the model was more uncertain about its misclassification on the images that are quite similar.

| image | y_true | prediction | y_pred | probability |
|:---:|:---:|:---:|:---:|:---:|
| 0 | Right-of-way at the next intersection | 0 | Right-of-way at the next intersection | 100.00% |
| 0 | Right-of-way at the next intersection | 1 | End of speed limit (80km/h) | 0.00% |
| 0 | Right-of-way at the next intersection | 2 | Roundabout mandatory | 0.00% |
| 0 | Right-of-way at the next intersection | 3 | Speed limit (80km/h) | 0.00% |
| 0 | Right-of-way at the next intersection | 4 | Pedestrians | 0.00% |
| 1 | Yield | 0 | Yield | 100.00% |
| 1 | Yield | 1 | Ahead only | 0.00% |
| 1 | Yield | 2 | No vehicles | 0.00% |
| 1 | Yield | 3 | No passing | 0.00% |
| 1 | Yield | 4 | Priority road | 0.00% |
| 2 | Yield | 0 | Yield | 100.00% |
| 2 | Yield | 1 | Ahead only | 0.00% |
| 2 | Yield | 2 | Priority road | 0.00% |
| 2 | Yield | 3 | No vehicles | 0.00% |
| 2 | Yield | 4 | No passing | 0.00% |
| 3 | Stop | 0 | Stop | 100.00% |
| 3 | Stop | 1 | Keep right | 0.00% |
| 3 | Stop | 2 | Speed limit (30km/h) | 0.00% |
| 3 | Stop | 3 | Priority road | 0.00% |
| 3 | Stop | 4 | Yield | 0.00% |
| 4 | General caution | 0 | General caution | 100.00% |
| 4 | General caution | 1 | Traffic signals | 0.00% |
| 4 | General caution | 2 | Road work | 0.00% |
| 4 | General caution | 3 | Road narrows on the right | 0.00% |
| 4 | General caution | 4 | Pedestrians | 0.00% |
| 5 | Speed limit (50km/h) | 0 | Speed limit (30km/h) | 64.13% |
| 5 | Speed limit (50km/h) | 1 | Speed limit (50km/h) | 35.87% |
| 5 | Speed limit (50km/h) | 2 | Speed limit (20km/h) | 0.00% |
| 5 | Speed limit (50km/h) | 3 | Speed limit (80km/h) | 0.00% |
| 5 | Speed limit (50km/h) | 4 | Roundabout mandatory  | 0.00% |
| 6 | Speed limit (50km/h) | 0 | Roundabout mandatory | 91.61% |
| 6 | Speed limit (50km/h) | 1 | Keep left | 3.78% |
| 6 | Speed limit (50km/h) | 2 | Speed limit (30km/h) | 2.96% |
| 6 | Speed limit (50km/h) | 3 | Speed limit (70km/h) | 0.58% |
| 6 | Speed limit (50km/h) | 4 | General caution | 0.45% |
| 7 | Bumpy road | 0 | Bumpy road | 100.00% | 
| 7 | Bumpy road | 1 | No vehicles | 0.00% | 
| 7 | Bumpy road | 2 | Turn left ahead | 0.00% | 
| 7 | Bumpy road | 3 | Yield | 0.00% | 
| 7 | Bumpy road | 4 | No passing | 0.00% | 
| 8 | Road work | 0 | Road work | 100.00% |
| 8 | Road work | 1 | Double curve | 0.00% |
| 8 | Road work | 2 | Wild animals crossing | 0.00% |
| 8 | Road work | 3 | Road narrows on the right | 0.00% |
| 8 | Road work | 4 | Speed limit (30km/h) | 0.00% |
| 9 | Children crossing | 0 | Bicycles crossing | 95.38% |
| 9 | Children crossing | 1 | Children crossing | 4.57% |
| 9 | Children crossing | 2 | Dangerous curve to the right | 0.05% |
| 9 | Children crossing | 3 | Road narrows on the right | 0.00% |
| 9 | Children crossing | 4 | General caution | 0.00% |
| 10 | Speed limit (70km/h) | 0 | Speed limit (70km/h) | 90.80% |
| 10 | Speed limit (70km/h) | 1 | Speed limit (20km/h) | 9.20% |
| 10 | Speed limit (70km/h) | 2 | Speed limit (30km/h) | 0.00% |
| 10 | Speed limit (70km/h) | 3 | Speed limit (80km/h) | 0.00% |
| 10 | Speed limit (70km/h) | 4 | Speed limit (100km/h) | 0.00% |