import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold

train = pd.read_csv('input/trainv3.csv')
test = pd.read_csv('input/testv3.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


#if __name__ == '__main__':
print 2
for column in list(train.select_dtypes(include=['object']).columns):
    if train[column].nunique() != test[column].nunique():
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)


        def filter_cat(x):
            if x in remove:
                return np.nan
            return x


        joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)

    joined[column] = pd.factorize(joined[column].values, sort=True)[0]

train = joined[joined['loss'].notnull()]
test = joined[joined['loss'].isnull()]

shift = 200
y = np.log(train['loss'] + shift)
ids = test['id']
ids_train = train['id']
X = train.drop(['loss', 'id'], 1)
X_test = test.drop(['loss', 'id'], 1)

RANDOM_STATE = 2016
params = {
    'min_child_weight': 1,
    'eta': 0.01,
    'eta_decay' : 0.9995,
    'colsample_bytree': 0.5,
    'max_depth': 12,
    'subsample': 0.8,
    'alpha': 1,
    'gamma': 1,
    'silent': 1,
    'verbose_eval': True,
    'seed': RANDOM_STATE
}

#xgtrain = xgb.DMatrix(X, label=y)
xgtest = xgb.DMatrix(X_test)

n_folds = 10
kf = KFold(X.shape[0], n_folds=n_folds, shuffle = True)
pred_test = 0
temp_cv_score = []

## CV loss data
cv_loss = pd.DataFrame(columns=["id","loss"])

for i, (train_index, test_index) in enumerate(kf):
    print "-" * 80
    print('\n Fold %d' % (i + 1))

    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgtrain_2 = xgb.DMatrix(X_val, label=y_val)

    watchlist = [(xgtrain, 'train'), (xgtrain_2, 'eval')]

    model = xgb.train(params, xgtrain, 5000, watchlist, feval=evalerror, verbose_eval=True, early_stopping_rounds= 300)
    pred_cv = np.exp(model.predict(xgtrain_2, ntree_limit=model.best_ntree_limit)) - shift
    pred_test += np.exp(model.predict(xgtest, ntree_limit=model.best_ntree_limit)) - shift

    temp_cv_score.append(mean_absolute_error(pred_cv, np.exp(y_val) - shift))
    cv_loss = pd.concat([cv_loss, pd.DataFrame({"id": ids_train[test_index], "loss": pred_cv})])

    print ('\n Fold %d' % (i + 1) + ' score: ' + str(temp_cv_score[i]))

finalscore = np.mean(temp_cv_score)
print('\n Final Score: ' + str(finalscore))
submission = pd.DataFrame()
submission['loss'] = pred_test/n_folds
submission['id'] = ids
name_string = 'submission_xgboost_' + str(n_folds) + '_' + \
               str(finalscore) + '.csv'
submission.to_csv(name_string, index=False)

name_string2 = "xgboost_CV_" + str(n_folds) + '_' + \
                '_' + str(finalscore) + '.csv'
cv_loss['id'] = cv_loss['id'].astype('int')
cv_loss.to_csv(name_string2, index = False)