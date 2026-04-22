import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model


"""
    Breif step-by-step process for simple ANN Workflow:-

        1. Initially import the pandas,numpy, tensorflow, Models-selection, text-preprocessing techinque and other things
        2. Then read the data using pandas code-script using the existing csv data file
        3. Drop irrelevant data from the data set(Cleaning Data) and Label any category coloum and transform it with fit_transform(data['Gender']) like this
        4. Choose another category of clm and convert them into vector form using "OneHotEncoder" or any other txt-preprocess (Data Conversion).
        5. Modify the Dataframe(df) with the latest chenge of "OneHotEncoder" values in the original df and drop the exist df of geography
        6. COnvert them into pickle file using pickle.dump(file-name, file)
        7. DiVide the dataset into indepent and dependent features
        8. ANN Implementation (Import the models, layers and callbacks)
        9. Train the model with Sequential
        10. Next add the optimizer(Adam), loss(binary_crossentropy) and then compile the model. refer:- model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        11. Once complied then store the logs in a seperate file and add the Earlystoping for it
        12. Train the model with X_train, X_test and Y_train, Y_test refer:- model.fit(
            X_train, y_train, validation_data = (X_test, y_test), epochs = 100,
            callbacks = [tensorflow_callback, early_stopping_callback]
            ) 
        13. Save the model and then check in the tensorflow dashboard how it's working.
    
    Once this is done we need to make predictions:-

        1. Load the trained model, scaler pickle,onehot
        2. Add some input data to check the predictions generation is correct or not
        3. Add the labeled data i.e modified data of "geography" to the original df
        4. Once it is done drop the old geography data and concat the transformed data
        5. Then do a model.predict() and validate the predictions

"""