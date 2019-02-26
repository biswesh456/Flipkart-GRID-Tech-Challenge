from os import path
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import Sequence
import json
import math

INPUT_PATH = "../input/edge-inputs-flipkart/new_input/new_input"
TEST_IMGS = path.join(INPUT_PATH, "images/test")
TEST_DATA = path.join(INPUT_PATH, "test.csv")
MODEL = "../input/flipkart/best_model.h5"
IMG_SIZE = (320, 240)
ORIG_IMG_SIZE = (640, 480)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 1)

class TestDataGenerator(Sequence):
    def __init__(self, test_df):
        self.avg_batch_size = int(len(test_df)**0.5)
        self.num_iter = int(math.ceil((len(test_df) * 1.) / self.avg_batch_size))
        self.image_names = test_df.image_name
    
    def __len__(self):
        return self.num_iter
    
    def __getitem__(self, idx):
        avg_batch_size = self.avg_batch_size
        num_iter = self.num_iter
        image_names = self.image_names
         
        temp = image_names.iloc[range(
            avg_batch_size * idx, 
            min(avg_batch_size * (idx+1), len(image_names))
        )]
        
        res = np.empty((0, IMG_SIZE[0], IMG_SIZE[1], 1))
        
        for img_name in temp:
            img_path = path.join(TEST_IMGS, img_name)
            img = image.load_img(img_path, target_size=IMG_SIZE, grayscale=True)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x /= 255.
            res = np.concatenate((res, x), 0)
        
        return res


def transform_bbox(df):
    conv_x = ((IMG_SIZE[0] * 1.) / ORIG_IMG_SIZE[0])
    conv_y = ((IMG_SIZE[1] * 1.) / ORIG_IMG_SIZE[1])
    
    df.w = df.w / conv_x
    df.h = df.h / conv_y
    df.x1 = df.x1 / conv_x
    df.y1 = df.y1 / conv_y
    
    df["x2"] = df.x1 + df.w
    df["y2"] = df.y1 + df.h
    
    df.x1 = df.x1.apply(lambda x : max(x, 0))
    df.y1 = df.y1.apply(lambda x : max(x, 0))
    df.x2 = df.x2.apply(lambda x : min(x, ORIG_IMG_SIZE[0]))
    df.y2 = df.y2.apply(lambda x : min(x, ORIG_IMG_SIZE[1]))
    
    return df.round(0).astype(int)
    

def get_predictions(model):
    test_df = pd.read_csv(TEST_DATA)
    
    # # Testing
    # test_df = test_df.iloc[range(5)]
    
    print("Loading test data...")
    test_gen = TestDataGenerator(test_df)
    print("Finsihed loading test data...")
    print()
    
    print("Predicting bounding box on test data...")
    bbox = model.predict_generator(test_gen)
    print("Finished predicting bounding box. Shape:", bbox.shape)
    print()
    
    bbox = pd.DataFrame(bbox, columns=["x1", "y1", "w", "h"])
    
    print("Transforming bounding box to required format...")
    bbox = transform_bbox(bbox)
    bbox["image_name"] = test_df.image_name
    print("Finished transforming bounding box...")
    print()
    
    print("Saving predictions...")
    bbox[["image_name", "x1", "x2", "y1", "y2"]].to_csv("./predictions_b.csv", index=False)
    
get_predictions(load_model(MODEL))