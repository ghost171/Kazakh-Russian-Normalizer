from os import name
from numpy.lib.function_base import append
import pandas as pd
from pandas.core.frame import DataFrame

df = pd.DataFrame()
ut_list = []
nt_list = []

for i in range(1750):
    ext = '.ut'
    name_ut = "sample_" + str("%0.7d"%i) + ext

    #list = df['unnormalized text'].append(data_folder_ut + name_ut)

    ut_list.append(name_ut)

    ext = '.nt'
    name_nt = "sample_" + "%0.7d"%i + ext
    data_folder_nt = 'normalized/'

    print("NAME_NT", "%0.7d"%i)

    nt_list.append(name_nt)
    #df_1 = pd.DataFrame({'unnormalized text' : [data_folder_ut + name_ut], 'normalized_text' : [data_folder_nt + name_nt]})

    #df.append(df_1, ignore_index=True)
    #print(df)

df['unnormalized'] = ut_list
df['normalized'] = nt_list

#compression_opts = dict(method='zip', archive_name='filelist.csv')  

print(df)

df.to_csv('filelist/filelist.csv', index=False, sep='\t', encoding='utf-8')