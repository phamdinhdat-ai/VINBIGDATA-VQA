import opendatasets as op 

url = "https://www.kaggle.com/datasets/lhanhsin/vizwiz"
#14a9a942523de92e16dcc880dc405b2a
op.download_kaggle_dataset(dataset_url=url, data_dir='.')
