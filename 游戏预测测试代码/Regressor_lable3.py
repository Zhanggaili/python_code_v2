
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np  # 用于数值计算
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import ElasticNet
import warnings
from sklearn.model_selection import train_test_split   # cross_validation
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
import gc
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

data_train = pd.read_csv("label_3.csv")
data_train = data_train.iloc[:,:109]
data_train_ = data_train.copy()
dt1=pd.to_datetime(data_train["register_time"])
data_train["register_time"] = dt1.dt.dayofyear
# data_train = data_train.drop("user_id",axis=1)
data_train['cha'] = data_train['prediction_pay_price']-data_train['pay_price']
data_train = data_train.drop(["lable",'prediction_pay_price'],axis=1)

data_train['cha_2'] = data_train['cha']
del data_train['cha']
data_train.rename(columns={'cha_2':'cha'}, inplace = True)
x_, y = data_train.iloc[:, 0:len(data_train.columns)-1], data_train.iloc[:,len(data_train.columns)-1:]
x = StandardScaler().fit_transform(x_)
x = pd.DataFrame(x)
x.columns = x_.columns

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

xgb = XGBRegressor(n_estimators=400, learning_rate=0.01,max_depth=4, random_state=1)
xgb.fit(x_train, y_train)
y_train_pred = xgb.predict(x_train)
y_test_pred = xgb.predict(x_test)
y_pred = xgb.predict(x)
print u'xgb：', mean_squared_error(y_train, y_train_pred)
print u'xgb：', mean_squared_error(y_test, y_test_pred)
print u'xgb：', mean_squared_error(y, y_pred)

data_test = pd.read_csv("test_label3.csv")
data_test = data_test.iloc[:,:108]
data_test_ = data_test.copy()
dt1=pd.to_datetime(data_test["register_time"])
data_test["register_time"] = dt1.dt.dayofyear
data_test.drop(["user_id"],axis=1, inplace=True)

test_pred_xgb = xgb.predict(data_test)
test_pred_ = data_test_.iloc[:,:1]
test_pred_['pre_xgb'] = test_pred_xgb
test_pred_['pre_xgb'] = test_pred_.apply(lambda x: 14.98 if (x['pre_xgb']>=11.09) else x['pre_xgb'] ,axis=1)
test_pred_['pre_xgb'] = test_pred_.apply(lambda x: 1.99 if (x['pre_xgb']<=3.0) else x['pre_xgb'] ,axis=1)
test_pred_['pay_price'] = data_test_['pay_price']
test_pred_['prediction_pay_price'] = test_pred_['pay_price']+test_pred_['pre_xgb']
test_pred_ = test_pred_.iloc[:,[0,3]]
test_pred_.to_csv('leibie3_0829.csv',index=None)


