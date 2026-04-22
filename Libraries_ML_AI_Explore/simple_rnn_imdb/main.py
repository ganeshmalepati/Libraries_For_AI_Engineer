# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

    

"""
    In the simple RNN, LSTM and GRU there is no connection of forward and backward of words. So, we are choosing the
    Bidirectional Recurrent Neural Networks (BRNNs) are an extension of traditional RNNs designed to process sequential data in both forward and backward directions. 
    This architecture allows the model to utilize both past and future context, making it particularly effective for tasks where understanding the entire sequence is crucial.

    How Bidirectional RNNs Work

    BRNNs consist of two RNN layers:

    A forward layer processes the sequence from start to end.
    A backward layer processes the sequence in reverse.
    At each time step, the outputs from both layers are concatenated to form a comprehensive representation. 
    This dual processing ensures that predictions are influenced by both past and future data.

    For example, in a sequence learning task, the forward hidden state at time t depends on the input at t and the previous hidden state, while the backward hidden state depends on the input at t and the next hidden state.

    
    Applications of Bidirectional RNNs

        BRNNs are widely used in:

        Sentiment Analysis: Understanding the sentiment of a sentence by considering both past and future context.

        Named Entity Recognition (NER): Identifying entities in text by analyzing bidirectional context.

        Machine Translation: Capturing the full context of source sentences for better translations.

        Speech Recognition: Improving transcription accuracy by considering both preceding and succeeding speech elements.
"""