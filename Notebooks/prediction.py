import preprocessing
import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout


cd = os.path.dirname(__file__)
data_directory = os.path.join(cd, '..', 'Data')

# Part 1: Create h5py file
with h5py.File(os.path.join(data_directory, 'data.h5'), 'w') as hdf_file:
    text_dataset = hdf_file.create_dataset('article_text', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    news_org_dataset = hdf_file.create_dataset('news_org', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    orientation_dataset = hdf_file.create_dataset('political_orientation', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    
    # Iterate over files
    for root, dirs, files in os.walk(os.path.join(data_directory, 'NELA_2017-2022')):
        for article in files:
            try:
                if article.endswith('.txt'):
                    article_path = os.path.join(root, article)
                    with open(article_path, 'r') as f:
                        article_text = f.read()
                    # Collect news organization name
                    news_org = str(preprocessing.get_news_org(article))
                    # Determine political orientation of news org
                    political_orientation = str(preprocessing.get_orientation_label(article))
                    # Append data to hdf5 datasets
                    text_dataset.resize((len(text_dataset) + 1, ))
                    text_dataset[-1] = article_text
                    news_org_dataset.resize((len(news_org_dataset) + 1, ))
                    news_org_dataset[-1] = news_org
                    orientation_dataset.resize((len(orientation_dataset) + 1, ))
                    orientation_dataset[-1] = political_orientation
            except Exception as e:
                print(f"Error processing file {article}: {str(e)}")
                
# Part 2: Training

with h5py.File(os.path.join(data_directory, 'data.h5'), 'r') as hdf_file:
    article_text = np.array(hdf_file['article_text'])
    political_orientation = np.array(hdf_file['political_orientation'])

# Represent data as DataFrame
df = pd.DataFrame({'Text': article_text, 'label': political_orientation})
df['Text'] = df['Text'].apply(lambda x: x.decode('utf-8'))
df['label'] = df['label'].apply(lambda x: x.decode('utf-8'))
df.tail()

# Encode labels

label_encoder = LabelEncoder()
df['labels_encoded'] = label_encoder.fit_transform(df['label'])

# Tokenize article text
max_words = 7000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['Text'])
X = tokenizer.texts_to_sequences(df['Text']) # X is our 
X = pad_sequences(X)

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, df['labels_encoded'], test_size=0.2, random_state=21)

# Construct LSTM
embedding_size = 100
lstm_out = 64

lstm_model = Sequential()
lstm_model.add(Embedding(max_words, embedding_size))
lstm_model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(7, activation='softmax')) # given that there are 7 unique classifications
lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Part 3: Train LSTM model on input text & political orientation labels

batch_size = 32
epochs = 5
history = lstm_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid), verbose=1)

'''There are a couple things I want to consider as the project runs on:
1. Addressing training overfitting
2. Resampling to address imbalance in the dataset labels
3. Additional model validation
4. Add 2018-2022 data
'''