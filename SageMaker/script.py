
import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#define model function
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return clf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #set SageMaker parameters
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train_features', type=str, default=os.environ['SM_CHANNEL_TRAIN_FEATURES'])
    parser.add_argument('--train_labels', type=str, default=os.environ['SM_CHANNEL_TRAIN_LABELS'])

    args = parser.parse_args()
    
    #load datasets
    X_train = pd.read_csv(os.path.join(args.train_features, 'wine_train_features.csv'), header=None)
    y_train = pd.read_csv(os.path.join(args.train_labels, 'wine_train_labels.csv'), header=None)

    #convert to NumPy arrays
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    #create pipeline with initial scaler
    model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    
    #train model
    model.fit(X_train, y_train)
    
    #save model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
