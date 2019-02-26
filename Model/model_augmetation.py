from os import path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import imgaug as ia
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import backend as K
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
import json

INPUT_PATH = "../input/edge-inputs-flipkart/new_input/new_input"
TRAIN_IMGS = path.join(INPUT_PATH, "images/train")
TRAIN_DATA = path.join(INPUT_PATH, "training.csv")
IMG_SIZE = (320, 240)
ORIG_IMG_SIZE = (640, 480)
BATCH_SIZE_TRAIN = 60
BATCH_SIZE_VAL = 37
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 1)
EPOCHS = 50

if K.image_data_format() == "channels_first":
    INPUT_SHAPE = (1, IMG_SIZE[0], IMG_SIZE[1])

def augmented_generator(img_gen):
    for x, y in img_gen:
        seq = iaa.Sequential([
            iaa.Sometimes(
                0.5,
                iaa.Affine(rotate=(-15, 15))
            ),
        ])
        
        bbs = []
        for i in range(y.shape[0]):
            bbs.append(ia.BoundingBoxesOnImage([
                    ia.BoundingBox(x1=y[i][0], y1=y[i][1], x2=y[i][0]+y[i][2], y2=y[i][1]+y[i][3]),
                ], 
                shape=x.shape[1:3])
            )

        seq_det = seq.to_deterministic()
        
        image_aug = seq_det.augment_images(x)
        bbs_aug = seq_det.augment_bounding_boxes(bbs)

        for i in range(len(bbs_aug)):
            after = bbs_aug[i].bounding_boxes[0]
            y[i][0] = after.x1
            y[i][1] = after.y1
            y[i][2] = after.x2-after.x1
            y[i][3] = after.y2-after.y1
        
        yield (x, y)


def image_data_gen(train_df, validate_df):
    train_gen = ImageDataGenerator(
        rescale=1. / 255,
    ).flow_from_dataframe(
        train_df,
        TRAIN_IMGS,
        x_col="image_name",
        y_col=["x1", "y1", "w", "h"],
        color_mode="grayscale",
        batch_size=BATCH_SIZE_TRAIN,
        target_size=IMG_SIZE,
        class_mode="other"
    )

    validate_gen = ImageDataGenerator(
        rescale=1. / 255,
    ).flow_from_dataframe(
        validate_df,
        TRAIN_IMGS,
        x_col="image_name",
        y_col=["x1", "y1", "w", "h"],
        color_mode="grayscale",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE_VAL,
        class_mode="other"
    )

    return train_gen, validate_gen


def model_cnn():
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = 3, activation="relu", input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters = 32, kernel_size = 3, activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters = 64, kernel_size = 3, activation="relu"))
    model.add(Conv2D(filters = 64, kernel_size = 3, activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters = 128, kernel_size = 3, activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(500, activation = "relu"))
    model.add(Dense(250, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4))

    return model


def transform_bbox(df):
    df["w"] = df.x2 - df.x1
    df["h"] = df.y2 - df.y1

    conv_x = ((IMG_SIZE[0] * 1.) / ORIG_IMG_SIZE[0])
    conv_y = ((IMG_SIZE[1] * 1.) / ORIG_IMG_SIZE[1])

    df.w = df.w * conv_x
    df.h = df.h * conv_y
    df.x1 = df.x1.apply(lambda x : max(x * conv_x, 0))
    df.y1 = df.y1.apply(lambda x : max(x * conv_y, 0))

    return df


def validate():
    df = transform_bbox(pd.read_csv(TRAIN_DATA))

    train_df, validate_df = train_test_split(
        df,
        train_size=0.9,
        random_state=200,
        shuffle=True
    )

    # ImageDataGenerator does not work if index doesn't start from 0
    train_df.reset_index(inplace=True, drop=True)
    validate_df.reset_index(inplace=True, drop=True)
    train_gen, validate_gen = image_data_gen(train_df, validate_df)
    train_gen = augmented_generator(train_gen)

    model = model_cnn()
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    print("Starting to train model...")
    hist = model.fit_generator(
        train_gen,
        epochs=EPOCHS,
        validation_data=validate_gen,
        validation_steps=len(validate_df) // BATCH_SIZE_VAL,
        steps_per_epoch=len(train_df) // BATCH_SIZE_TRAIN,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                './best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True
            )
        ]
    )
    print("Training complete...")

    print("Saving history...")
    with open("./history.json", "w") as f:
        json.dump(hist.history, f)
    print("Saved history...")

    print("Saving model summary...")
    with open("./model.txt", "w") as f:
        model.summary(model.summary(print_fn=lambda x: f.write(x + "\n")))
    print("Saved model summary...")

validate()
