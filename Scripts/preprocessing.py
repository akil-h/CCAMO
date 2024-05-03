import os
import pandas as pd
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Get directory of current script
cd = os.path.dirname(__file__)

# Create relative path for master sheet
master_sheet_file_path = '/Users/akilhuang/Documents/Projects/CCAMO/Data/Updated NELA Master Spreadsheet.xlsx' # os.path.join(cd, '..', 'Data', 'Updated NELA Master Spreadsheet.xlsx')

# Read master sheet containing political orientation mapping
try:
    mapping_df = pd.read_excel(master_sheet_file_path, sheet_name='Master Sheet', engine='openpyxl')
except FileNotFoundError:
    print(f"Error: The file at '{master_sheet_file_path}' was not found.")

# Create a dictionary mapping news organizations to their respective political orientation labels. 
news_org_to_orientation = {
    org.replace(' ', '').lower(): orientation
    for org, orientation in zip(mapping_df['source (Master List)'], mapping_df['2022 Media Bias/Fact Check'])
}


def get_news_org(file_name: str) -> str:
    '''This helper function returns name of article news organization in lowercase.
    
    Example:
>>> get_news_org("CNN--article.txt")
    'cnn'
    '''
    news_org = file_name[:file_name.index('-')].replace(' ', '').lower() if '-' in file_name else file_name.replace(' ', '').lower()
    return news_org


def get_orientation_label(file_name: str) -> str:
    '''Returns the political orientation of the news organization to which an article belongs to.
    
    Example:
>>> get_orientation_label("CNN--article.txt")
    'left-center'
    '''
    # Retrieve the political orientation label from the dictionary
    news_org = get_news_org(file_name)
    try:
        orientation = news_org_to_orientation[news_org]
    except KeyError:
        raise KeyError(f"Political orientation label not found for org: {news_org}")
    # Note: Some news organizations do not have labeled political orientation labels

    return orientation


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text):
    # Tokenize text
    tokens = nltk.word_tokenize(text.lower()) # list of individual tokens
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word.isalnum() and not word in stop_words]
    # Stem
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)