import preprocessing
import os
import h5py
import re
import pandas as pd

cd = os.path.dirname(__file__)
data_path = os.path.join(cd, '..', 'Data', 'NELA_2017-2022')

# Part 1: Create h5py file
with h5py.File('data.h5', 'w') as hdf_file:
    text_dataset = hdf_file.create_dataset('article_text', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    news_org_dataset = hdf_file.create_dataset('news_org', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    orientation_dataset = hdf_file.create_dataset('political_orientation', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    
    # Iterate over files
    for root, dirs, files in os.walk(data_path):
        for file in files:
            try:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        article_text = f.read()
                    # Collect news organization name
                    news_org = preprocessing.get_news_org(file)
                    # Determine political orientation of news org
                    political_orientation = preprocessing.get_orientation_label(file)
                    # Append data to hdf5 datasets
                    text_dataset.resize((len(text_dataset) + 1, ))
                    text_dataset[-1] = article_text
                    news_org_dataset.resize((len(news_org_dataset) + 1, ))
                    news_org_dataset[-1] = news_org
                    orientation_dataset.resize((len(orientation_dataset) + 1, ))
                    orientation_dataset[-1] = political_orientation
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                
# Part 2: Split data



# Train CNN/SVM/GBM on input text & political orientation labels

