from modules import *
#import zipfile
#import pandas as pd

#loading and unzipping the dataset
def unzipping(source, destination):
    #source = 'C:\Users\Sayan Hazra\Temi\Galaxy classification'
    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(destination)

# Creating the dataframe for galaxy label segmentation
def label_dataframe(label_data):
    df = pd.read_csv(label_data)
    print('Galaxy classification dataframe: \n',df.head)
    print('\nDataframe shape: ',df.shape)
    return df

    
