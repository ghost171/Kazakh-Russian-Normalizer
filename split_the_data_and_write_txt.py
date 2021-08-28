import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('out_final.tsv', sep='\t')

print(df.columns)

X = df.drop(columns = ['Әлеуметтік желілерде ЕЦ-жүз алпыс алты/бес төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.']).copy()
y = df['Әлеуметтік желілерде ЕЦ-жүз алпыс алты/бес төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.']

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)

test_size=0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)

print('______________________________________________')
print("X_train", X_train)
print('______________________________________________')

X_train = X_train['Әлеуметтік желілерде ЕЦ-166/5 төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.'].to_list()
X_train = X_train
print(y_train)
y_train = y_train.to_list()
y_train = y_train

out_train = pd.DataFrame()
out_train['Әлеуметтік желілерде ЕЦ-166/5 төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.'] = X_train
np.savetxt(r'train.ut', out_train.values, fmt='%s')

out_train = pd.DataFrame()
out_train['Әлеуметтік желілерде ЕЦ-жүз алпыс алты/бес төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.'] = y_train
np.savetxt(r'train.nt', out_train.values, fmt='%s')



'''print("X", X['samples'])
X_help = X['samples'].to_list()
for i in range(1024, len(X_help), 1024):
    X_1 = X_help[i - 1024:i]
    out_train = pd.DataFrame()
    
    out_train['SRC'] = X_1
    name = 'data_' + str(int(i / 1024)) + '.ut'
    np.savetxt(r'' + name, out_train.values, fmt='%s')


y_help = y.to_list()
for i in range(1024, len(y_help), 1024):
    y_1 = y_help[i - 1024:i]
    out_train = pd.DataFrame()
    out_train['TRG'] = y_1
    name = 'data_' + str(int(i / 1024)) + '.nt'
    np.savetxt(r'' + name, out_train.values, fmt='%s')'''

X_valid = X_valid['Әлеуметтік желілерде ЕЦ-166/5 төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.'].to_list()
X_valid = X_valid

y_valid = y_valid.to_list()
y_valid = y_valid

out_valid = pd.DataFrame()
out_valid['Әлеуметтік желілерде ЕЦ-166/5 төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.'] = X_valid
np.savetxt(r'valid.ut', out_valid.values, fmt='%s')

out_valid = pd.DataFrame()
out_valid['Әлеуметтік желілерде ЕЦ-жүз алпыс алты/бес төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.'] = y_valid
np.savetxt(r'valid.nt', out_valid.values, fmt='%s')

X_test = X_test['Әлеуметтік желілерде ЕЦ-166/5 төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.'].to_list()
X_test = X_test

y_test = y_test.to_list()
y_test = y_test

out_test = pd.DataFrame()
out_test['Әлеуметтік желілерде ЕЦ-166/5 төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.'] = X_test
np.savetxt(r'test.ut', out_test.values, fmt='%s')

out_test = pd.DataFrame()
out_test['Әлеуметтік желілерде ЕЦ-жүз алпыс алты/бес төтенше қауіпсіздік мекемесінде тұтқындардың жаппай төбелесінен кейін әскер енгізіліп, тінту жүргізілгені туралы ақпарат тарады.'] = y_test
np.savetxt(r'test.nt', out_test.values, fmt='%s')