import os
import pandas as pd

# Get directory of current script
cd = os.path.dirname(__file__)

# Create relative path for master sheet
master_sheet_file_path = os.path.join(cd, '..', 'Data', '~$Updated NELA Master Spreadsheet - June 22.xlsx')

# Read master sheet containing political orientation mapping
try:
    mapping_df = pd.read_excel(master_sheet_file_path, sheet_name='Master Sheet')
except FileNotFoundError:
    print(f"Error: The file at '{master_sheet_file_path}' was not found.")


def get_news_org(file_name: str) -> str:
    '''Returns name of article news organization.
    
    Example:
>>> get_news_org("CNN--article.txt")
    'CNN'
    '''
    return file_name[:file_name.index('-')].strip() if '-' in file_name else file_name


def get_orientation_label(file_name: str) -> str:
    '''Returns the political orientation of the news organization to which an article belongs to.
    
    Example:
>>> get_orientation_label("CNN--article.txt")
    'left-center'
    '''
    # Create a dictionary mapping news organizations to their respective political orientation labels. 
    news_org_to_orientation = dict(zip(mapping_df['source (Master List)'], mapping_df['2022 Media Bias/Fact Check']))
    # Retrieve the political orientation label from the dictionary
    news_org = get_news_org(file_name)
    try:
        orientation = news_org_to_orientation[news_org]
    except KeyError:
        raise KeyError(f"Political orientation label not found for org: {news_org}")
    # Note: Some news organizations do not have labeled political orientation labels

    return orientation