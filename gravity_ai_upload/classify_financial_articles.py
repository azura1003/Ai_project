from gravityai import gravityai as grav
import pickle
import pandas as pd

model = pickle.load(open(''))
tfid_vectorizer = pickle.load(open(''))
label_encoder = pickle.load(open(''))


def process(inPath, outPath):
    # read input file
    input_df = pd.read_csv(inPath)
    # vectorize the data
    features = tfid_vectorizer.transform(input_df['body'])
    # predict the classes
    predictions = model.predict(features)
    # convert output labels to categories
    input_df['category'] = label_encoder.inverse_transform(predictions)
