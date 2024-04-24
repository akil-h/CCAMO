import os
import h5py

# 1. Create h5py file
with h5py.File('data.h5', 'w') as hdf_file:
    text_dataset = hdf_file.create_dataset('articles', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    new_org_dataset = hdf_file.create_dataset('news_orgs', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))

    # Iterate over files

# 2. Split data

# 3. Train CNN/SVM/GBM on input text & political orientation labels

