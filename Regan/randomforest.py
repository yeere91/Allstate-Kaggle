import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

print "Loading Data..."
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
print train.head()
print test.head()

print "Training set is of " +  str(len(train)) + " length"
print "Training set is of " +  str(len(test)) + " length"

ids = test['id']
ids_train = train['id']
X = train.drop(['loss', 'id'], 1)

#len(train) * .8
#train['loss']

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

print '-' *50
print "Random Forest Model"
print '-' *50

## n_jobs = -1: Run the job with the number of cores on your computer
## n_estimators = 50: Run a random forest with 50 trees
## random_state: seed for the state of start of job
model = RandomForestRegressor(n_jobs=-1,n_estimators=50,random_state=seed)
print model

model.fit()
#result = mean_absolute_error(np.expm1(Y_val), np.expm1(model.predict(X_val[:,i_cols_list])))
#print result
