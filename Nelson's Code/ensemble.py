import pandas as pd

submission = pd.read_csv('submission_keras_shift_perm_10_10_60.csv')
submission['loss'] *= 0.5

submission['loss'] += 0.25 * pd.read_csv('sub_v.csv')['loss'].values
# prediction from https://www.kaggle.com/mariusbo/allstate-claims-severity/lexical-encoding-feature-comb-lb-1109-05787
submission['loss'] += 0.25 * pd.read_csv('submission_keras_shift_perm_10_10_30.csv')['loss'].values
# new prediction with score 1105 on LB
submission.to_csv('sub_lb_1105.csv', index=False)