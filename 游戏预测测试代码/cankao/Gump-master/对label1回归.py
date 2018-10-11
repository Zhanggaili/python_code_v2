
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 180)
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # 读入数据

# In[2]:


df = pd.read_csv("tap_fun_train.csv",encoding="utf-8")
df = df.rename(columns={"treatment_acceleraion_add_value":"treatment_acceleration_add_value","sr_rss_a_prod_levell":"sr_rss_a_prod_level"})
df.head()


# # 取出为1的

# In[4]:


display("総数:{}".format(df.shape[0]))
display("陽性:{}".format(df[df["pay_price"] != df["prediction_pay_price"]].shape[0]))

df["label"]=0
df.loc[df["pay_price"] != df["prediction_pay_price"],"label"]=1
df_positive = df[df["label"]==1].copy()


# In[19]:


Y_df_positive = df_positive["prediction_pay_price"] 
X_df_positive = df_positive.drop(columns=["user_id","register_time","prediction_pay_price","label"])


# In[22]:


X_train, X_val, Y_train, Y_val = train_test_split(X_df_positive, Y_df_positive, test_size = 0.2)


# # 回归

# In[28]:


mod = xgb.XGBRegressor()
#mod = linear_model.LinearRegression()

mod.fit(X_train, Y_train)

Y_train_pred = mod.predict(X_train)
Y_val_pred = mod.predict(X_val)


# # RMSE & R^2 score

# In[34]:


print('RMSE train : %.3f, test : %.3f' % (np.sqrt(mean_squared_error(Y_train, Y_train_pred)), np.sqrt(mean_squared_error(Y_val, Y_val_pred))) )
print('R^2 train : %.3f, test : %.3f' % (r2_score(Y_train, Y_train_pred), r2_score(Y_val, Y_val_pred)) )


# # 残差

# In[35]:


plt.figure(figsize = (10, 7))
plt.scatter(Y_train_pred, Y_train_pred - Y_train, c = 'black', marker = 'o', s = 35, alpha = 0.5, label = 'Training data')
plt.scatter(Y_val_pred, Y_val_pred - Y_val, c = 'lightgreen', marker = 's', s = 35, alpha = 0.7, label = 'Val data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([-10, 50])
plt.show()


# # grid search

# In[47]:


# グリッドサーチに必要なクラスのインポート
from sklearn.grid_search import GridSearchCV
# サーチするパラメータは範囲を指定
params = {'max_depth': [3, 5, 10], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 10, 100], 'subsample': [0.8, 0.85, 0.9, 0.95], 'colsample_bytree': [0.5, 1.0]}
# モデルのインスタンス作成
mod = xgb.XGBRegressor()
# 10-fold Cross Validationでパラメータ選定
cv = GridSearchCV(mod, params, cv = 10, scoring= 'neg_mean_squared_error', n_jobs =1)
cv.fit(X_train, Y_train)



# In[ ]:


Y_train_pred = cv.predict(X_train)
Y_val_pred = cv.predict(X_val)


# In[ ]:


print('RMSE train : %.3f, test : %.3f' % (np.sqrt(mean_squared_error(Y_train, Y_train_pred)), np.sqrt(mean_squared_error(Y_val, Y_val_pred))) )
print('R^2 train : %.3f, test : %.3f' % (r2_score(Y_train, Y_train_pred), r2_score(Y_val, Y_val_pred)) )


# In[ ]:


plt.figure(figsize = (10, 7))
plt.scatter(Y_train_pred, Y_train_pred - Y_train, c = 'black', marker = 'o', s = 35, alpha = 0.5, label = 'Training data')
plt.scatter(Y_val_pred, Y_val_pred - Y_val, c = 'lightgreen', marker = 's', s = 35, alpha = 0.7, label = 'Val data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([-10, 50])
plt.show()

