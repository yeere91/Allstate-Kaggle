# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy

#range of features considered
split = 116 
#number of features considered
size = 15

print "Reading dataset..."
#Read the train dataset
dataset = pd.read_csv('../data/train.csv')
#Read test dataset
dataset_test = pd.read_csv('../data/test.csv')

#Save the id's for submission file
ID = dataset_test['id']
#Drop unnecessary columns
dataset_test.drop('id',axis=1,inplace=True)

#Print all rows and columns. Dont hide any
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#Drop the first column 'id' since it just has serial numbers. Not useful in the prediction process.
dataset = dataset.iloc[:,1:]

#create a dataframe with only continuous features
data=dataset.iloc[:,split:] 

#names of all the columns
cols = dataset.columns


print "Encoding Data..."
#Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#Variable to hold the list of variables for an attribute in the train and test data
labels = []

for i in range(0,split):
    train = dataset[cols[i]].unique()
    test = dataset_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))    

del dataset_test

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset.iloc[:,i])
    feature = feature.reshape(dataset.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

# Print the shape of the encoded data
print "Encoded Data Shape:"
print(encoded_cats.shape)

#Concatenate encoded attributes with continuous attributes
dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)
del cats
del feature
del dataset
del encoded_cats
print "Reattaching continuous features..."
print(dataset_encoded.shape)

print "Creating Training and Validation sets..."
#get the number of rows and columns
r, c = dataset_encoded.shape

#create an array which has indexes of columns
i_cols = []
for i in range(0,c-1):
    i_cols.append(i)
print i_cols

#Y is the target column, X has the rest
X = dataset_encoded[:,0:(c-1)]
Y = dataset_encoded[:,(c-1)]
del dataset_encoded

X=pd.DataFrame(X)
Y=pd.DataFrame(Y)
X.to_csv("EncodedTrain.csv")
Y.to_csv("EncodedLosses.csv")
#Validation chunk size
val_size = 0.1

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

#Split the data into chunks
from sklearn import cross_validation
from sklearn.cross_validation import KFold
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)


n_folds = 10
kf = KFold(X.shape[0], n_folds=n_folds, shuffle = True)
pred_test = 0
temp_cv_score = []

## CV loss data
cv_loss = pd.DataFrame(columns=["id","loss"])

for i, (train_index, test_index) in enumerate(kf):
    print i
    print "-" * 80
    print('\n Fold %d' % (i + 1))

    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_val = Y.iloc[train_index], Y.iloc[test_index]
    print "XTRAIN:"
    print X_train
    print "YTRAIN:"
    print Y_train
# #All features
# X_all = []

# #List of combinations
# comb = []

# #Dictionary to store the MAE for all algorithms 
# mae = []

# #Scoring parameter
# from sklearn.metrics import mean_absolute_error

# #Add this version of X to the list 
# n = "All"
# #X_all.append([n, X_train,X_val,i_cols])
# X_all.append([n, i_cols])
# print X_all

print "Random Forest Algo:"
#Evaluation of various combinations of RandomForest

#Import the library
from sklearn.ensemble import RandomForestRegressor

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([50])

algo = "RF"
print algo

for n_estimators in n_list:
    #Set the base model
    model = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
    model.fit(X_train[:,1],Y_train)
    
    
print numpy.expm1(model.predict(X_val[:,i_cols_list]))
        
#     comb.append(algo + " %s" % n_estimators )

print mae
print "Done!"

# if (len(n_list)==0):
#     mae.append(1213)
#     comb.append("RF" + " %s" % 50 )    
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is 1213 when the number of estimators is 50

