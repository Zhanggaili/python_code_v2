
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 180)
from sklearn.externals import joblib


# In[67]:


df = pd.read_csv("tap_fun_test.csv",encoding="utf-8")
df = df.rename(columns={"treatment_acceleraion_add_value":"treatment_acceleration_add_value","sr_rss_a_prod_levell":"sr_rss_a_prod_level"})


# In[68]:


display("総数:{}".format(df.shape[0]))


# In[69]:


df.head()


# In[115]:


X_df = df.drop(columns=["register_time"])


# In[116]:


clf = joblib.load("binary_classifier_model.sav")


# In[117]:


preds = clf.predict_proba(X_df.drop(columns=["user_id"]))[:,1]


# In[118]:


Y_pred =  np.where(preds<=0.5, 0, 1)


# In[119]:


X_df["label"] = Y_pred


# In[120]:


X_df["prediction_pay_price"] = 0


# In[121]:


X_df["prediction_pay_price"] = X_df.apply(lambda row: row["pay_price"]  if row["label"]==0 else 0 , axis=1)


# In[122]:


X_df_positive = X_df[X_df["label"]==1]


# In[123]:


X_df_positive.head()


# In[124]:


mod = joblib.load("regression_model.sav")


# In[125]:


Y_df_positive = mod.predict(X_df_positive.drop(columns=["user_id","prediction_pay_price","label"]))


# In[126]:


Y_df_positive


# In[127]:


#15403
X_df.loc[X_df_positive.index.tolist(), 'prediction_pay_price']=Y_df_positive


# In[128]:


X_df[X_df["user_id"]==15403]


# In[129]:


result=X_df[["user_id","prediction_pay_price"]]


# In[130]:


result


# In[131]:


result.to_csv("result2.csv",encoding="utf-8",index=False)

