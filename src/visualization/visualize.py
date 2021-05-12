# python core packages
import random

# 3rd party packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def random_images(X_train, y_train, n_train, class_names, n_columns = 6, n_rows = 3):

    # generate n random integers from the trainingsset
    integers = [random.randint(0, n_train) for p in range(0, (n_columns * n_rows))]

    # show the images
    plt.figure(figsize=(22, 10))
    for index, integer in enumerate(integers):
        class_id = y_train[integer]
        
        ax = plt.subplot(n_rows, n_columns, index + 1)
        plt.imshow(X_train[integer])
        plt.title(f'{class_names[class_id]} ({class_id})')
        plt.axis("off")


def class_distribution(y_train, class_names, threshold=1000):
    # count per value
    values, counts = np.unique(y_train, return_counts=True)

    # map the values to labels
    labels = [class_names[value] for value in values]

    # create a dataframe (so i can sort) and then plot the value
    df = pd.DataFrame({'labels':labels, 'counts':counts}).sort_values('counts', ascending=True).set_index('labels')

    # define a subplot
    fig, ax = plt.subplots(figsize=(20, 15))
    
    # plot the bars
    bars = ax.barh(df.index, df['counts'])
    
    # draw the vertical threshold line
    ax.axvline(x=threshold, color='red', linewidth=0.8, linestyle="--", label='threshold')

    # remove the borders
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # loop over the bars
    for bar in bars:
        # get the bar value
        label = bar.get_width()
        
        # determine the label y position
        label_y_pos = bar.get_y() + bar.get_height() / 2
        
        # add the label
        ax.text(label, label_y_pos, s=f'{label:.0f}', va='center', ha='right', fontsize=15, color='white')
        
        # color the bars 
        if label > threshold:
            bar.set_color('green')
        else:
            bar.set_color('orange')

    plt.title('Nr training images per class')
    plt.xlabel('Nr images')
    plt.ylabel('Class label')
    plt.legend()


def history(history, threshold):
    plt.figure(figsize=(22, 10))

    # summarize history for accuracy
    plt.subplot(3, 1, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.axhline(y=threshold, color='red', linewidth=0.8, linestyle="--", label='threshold')
    plt.legend(loc='lower right')
    plt.title('Training & Validation history')
    plt.ylabel('Accuracy')
    plt.ylim([0,1.0])

    # summarize history for loss
    plt.subplot(3, 1, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.ylim([0,1.0])

    # summarize history for learning rate
    plt.subplot(3, 1, 3)
    plt.plot(history.history['lr'])
    plt.ylabel('Learning rate')
    plt.xlabel('Epoch')


def evaluations(df, threshold = 0.9):
    # define a subplot
    fig, ax = plt.subplots(figsize=(20, 5))
    
    # plot the bars
    bars = ax.barh(df.index, df['accuracy'])
    
    # draw the vertical threshold line
    ax.axvline(x=threshold, color='red', linewidth=0.8, linestyle="--", label='threshold')

    # remove the borders
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # loop over the bars
    for bar in bars:
        # get the bar value
        label = bar.get_width()
        
        # determine the label y position
        label_y_pos = bar.get_y() + bar.get_height() / 2
        
        # add the label
        ax.text(label, label_y_pos, s=f'{label:.3f}', va='center', ha='right', fontsize=15, color='white')
        
        # color the bars 
        if label > threshold:
            bar.set_color('green')
        else:
            bar.set_color('orange')

    plt.title('Accuracy per dataset')
    plt.xlabel('Accuracy')
    plt.ylabel('Dataset')
    plt.legend()