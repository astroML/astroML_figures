# Generate a CNN for classifying SDSS galaxy images using the catalogs of
# Nair and Abraham (2010) http://adsabs.harvard.edu/abs/2010ApJS..186..427N
# Ellipticals are class 0. Spirals are class 1
# Derived from https://github.com/mhuertascompany/IAC_XXX_WINTER (Marc Huertas Company)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import roc_curve

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

import random

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)
plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05


def read_savefile(filename):
    '''Read npy save file containing images or labels of galaxies'''
    return np.load(filename)


def CNN(img_channels, img_rows, img_cols, verbose=False):
    '''Define CNN model for Nair and Abraham data'''

    # some hyperparamters you can chage
    dropoutpar = 0.5
    nb_dense = 64

    model = Sequential()
    model.add(Convolution2D(32, 6, 6, border_mode='same',
                            input_shape=(img_rows, img_cols, img_channels)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(nb_dense, activation='relu'))
    model.add(Dropout(dropoutpar))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    print("Compilation...")

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    print("... done!")
    if verbose is True:
        print("Model Summary")
        print("===================")
        model.summary()
    return model


def train_CNN(X, Y, ntrain, nval, output="test", verbose=False):
    '''Train the CNN given a dataset and output model and weights'''

    # train params - hardcoded for simplicity
    batch_size = 30
    nb_epoch = 50
    data_augmentation = True  # if True the data will be augmented at every iteration

    ind = random.sample(range(0, ntrain+nval-1), ntrain+nval-1)
    X_train = X[ind[0:ntrain], :, :, :]
    X_val = X[ind[ntrain:ntrain+nval], :, :, :]
    Y_train = Y[ind[0:ntrain]]
    Y_val = Y[ind[ntrain:ntrain+nval]]

    # input image dimensions
    img_rows, img_cols = X_train.shape[1:3]
    img_channels = 3

    # Right shape for X
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,
                              img_channels)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, img_channels)

    # Avoid more iterations once convergence
    patience_par = 10
    earlystopping = EarlyStopping(monitor='val_loss', patience=patience_par,
                                  verbose=0, mode='auto' )
    modelcheckpoint = ModelCheckpoint(output+"_best.hd5", monitor='val_loss',
                                      verbose=0, save_best_only=True)

    # Define CNN
    model = CNN(img_channels, img_rows, img_cols, verbose=True)

    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, Y_val),
                            shuffle=True, verbose=verbose,
                            callbacks=[earlystopping, modelcheckpoint])
    else:
        print('Using real-time data augmentation.')
        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=45,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=[0.75, 1.3])

        datagen.fit(X_train)

        history = model.fit_generator(
            datagen.flow(X_train, Y_train, batch_size=batch_size),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_epoch,
            validation_data=(X_val, Y_val),
            callbacks=[earlystopping, modelcheckpoint])

    print("Saving model...")
    # save weights
    model.save_weights(output+".weights", overwrite=True)


def apply_CNN(X, model_name):
    '''Apply a CNN to a data set'''
    # input image dimensions
    img_rows, img_cols = X.shape[1:3]
    img_channels = 3
    X = X.reshape(X.shape[0], img_rows, img_cols, img_channels)

    # load model & predict
    print("Loading weights", model_name)

    model = CNN(img_channels, img_rows, img_cols)
    model.load_weights(model_name+".weights")
    Y_pred = model.predict_proba(X)

    return Y_pred


def add_titlebox(ax, text):
    '''Add an embedded title into figure panel'''
    ax.text(.1, .85, text,
            horizontalalignment='left',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    return ax


def plot_CNN_performance(pred, labels):
    '''Plot ROC curve and sample galaxies'''

    fig = plt.figure(figsize=(6, 3))
    fig.subplots_adjust(wspace=0.1, hspace=0.1,
                        left=0.1, right=0.95,
                        bottom=0.15, top=0.9)

    # define shape of figure
    gridsize = (2, 4)
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid(gridsize, (0, 2))
    ax3 = plt.subplot2grid(gridsize, (0, 3))
    ax4 = plt.subplot2grid(gridsize, (1, 2))
    ax5 = plt.subplot2grid(gridsize, (1, 3))

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(labels, pred)

    ax1.plot(fpr, tpr, color='black')
    ax1.set_xlabel(r'False Positive Rate')
    ax1.set_ylabel(r'True Positive Rate')

    # array of objects (good E, good S, bad E, bad S)
    goodE = np.where((pred[:, 0] < 0.5) & (labels == 0))
    goodS = np.where((pred[:, 0] > 0.5) & (labels == 1))
    badE = np.where((pred[:, 0] < 0.5) & (labels == 1))
    badS = np.where((pred[:, 0] > 0.5) & (labels == 0))

    ax2.imshow(D[pred_index + goodE[0][1]])
    add_titlebox(ax2, "Correct E")
    ax2.axis('off')

    ax3.imshow(D[pred_index + goodS[0][4]])
    add_titlebox(ax3, "Correct Spiral")
    ax3.axis('off')

    ax4.imshow(D[pred_index + badE[0][1]])
    add_titlebox(ax4, "Incorrect E")
    ax4.axis('off')

    ax5.imshow(D[pred_index + badS[0][3]])
    add_titlebox(ax5, "Incorrect Spiral")
    ax5.axis('off')

    plt.show()


n_objects = 500
save_files = "./SDSS{}".format(n_objects)

# Read SDSS images and labels
D = read_savefile("sdss_images_1000.npy")[0:n_objects]
Y = read_savefile("sdss_labels_1000.npy")[0:n_objects]

# Train network and output to disk (keep 10% of data for test set)
ntrain = D.shape[0] * 8 // 10.
nval = D.shape[0] // 10
npred = D.shape[0] - (ntrain + nval)  # test sample size;
pred_index = ntrain + nval            # test sample start index;

# Normalize images
mu = np.amax(D, axis=(1, 2))
for i in range(0, mu.shape[0]):
    D[i, :, :, 0] = D[i, :, :, 0] / mu[i, 0]
    D[i, :, :, 1] = D[i, :, :, 1] / mu[i, 1]
    D[i, :, :, 2] = D[i, :, :, 2] / mu[i, 2]

# change order so that we do not use always the same objects to train/test
D, Y, = shuffle(D, Y, random_state=0)

my_file = Path(save_files + ".weights")
if my_file.is_file():
    Y_pred = apply_CNN(D[pred_index:pred_index + npred, :, :, :], save_files)
    Y_test=Y[pred_index:pred_index + npred]
else:
    print("Training Model")
    print("====================")
    model_name = train_CNN(D, Y, ntrain, nval, output=save_files)
    Y_pred = apply_CNN(D[pred_index:pred_index + npred, :, :, :], save_files)
    Y_test = Y[pred_index:pred_index + npred]

Y_pred_class = Y_pred * 0
Y_pred_class[Y_pred > 0.5] = 1
print("Global Accuracy:", accuracy_score(Y_test, Y_pred_class))


plot_CNN_performance(Y_pred, Y_test)
