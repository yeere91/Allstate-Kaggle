# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np



print ("Reading dataset...")
#Read the train dataset
dataset = pd.read_csv('../input/trainv4.csv')
#Read test dataset
dataset_test = pd.read_csv('../input/testv4.csv')


## cat117 is cont 2
## cat 118 is High vs. Low
print ("Factorizing categorical variables...")
features = dataset.columns
cats = [feat for feat in features if 'cat' in feat]
print cats
for feat in cats:
    dataset[feat] = pd.factorize(dataset[feat], sort=True)[0]



print "Creating Training and Validation sets..."
#get the number of rows and columns
r, c = dataset.shape
print r
print c

index = list(dataset.index)
np.random.shuffle(index)
print index[0:10]
train = dataset.iloc[index]

## response and IDs
response = np.log(train['loss'].values + 200)
print response

id_train = train['id'].values
print id_train

id_test = dataset_test['id'].values


print("-")*50
print "Random Forest Algo"
print("-")*50
#Evaluation of various combinations of RandomForest

from sklearn.ensemble import RandomForestRegressor
seed = 0

print "Training Random Forest Model..."
model = RandomForestRegressor(n_jobs=-1,n_estimators=50,random_state=seed)
model.fit(train, response)
