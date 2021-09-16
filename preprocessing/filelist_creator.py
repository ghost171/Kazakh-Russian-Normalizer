import pathlib
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

filepaths = pathlib.Path('data/raw/unnormalized').rglob('*.txt')
filenames = [ str(f).replace('data/raw/unnormalized/', '') for f in filepaths]
print(len(filenames))

df = pd.DataFrame({'filenames': filenames})

inds = list(np.arange(len(filenames)))
inds_train_val, inds_test = train_test_split(inds, test_size = 20000, random_state = 42)
inds_train, inds_val = train_test_split(inds, test_size = 100000, random_state = 42)

df_train = df.loc[inds_train]
df_val = df.loc[inds_val]
df_test = df.loc[inds_test]

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)

df_train.to_csv('filelists/train_filelists.csv', index = False, header = False)
df_val.to_csv('filelists/val_filelists.csv', index = False, header = False)
df_test.to_csv('filelists/test_filelists.csv', index = False, header = False)
