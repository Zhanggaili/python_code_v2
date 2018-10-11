
# coding: utf-8

# # Random Forest

# In[1]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 180)
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# In[69]:


df = pd.read_csv("tap_fun_train.csv",encoding="utf-8")
df = df.rename(columns={"treatment_acceleraion_add_value":"treatment_acceleration_add_value","sr_rss_a_prod_levell":"sr_rss_a_prod_level"})
df.head()


# In[154]:


display("総数:{}".format(df.shape[0]))
display("陽性:{}".format(df[df["pay_price"] != df["prediction_pay_price"]].shape[0]))


# In[155]:


df["label"]=0


# In[156]:


df.loc[df["pay_price"] != df["prediction_pay_price"],"label"]=1


# In[157]:


df_positive = df[df["label"]==1].copy()
df_negative = df[df["label"]==0].copy()


# In[158]:


# 按比例分割
def split_train_val (df_positive, df_negative, val_size=0.2):
    df_positive_train, df_positive_val = train_test_split(df_positive, test_size = 0.20)
    df_negative_train, df_negative_val = train_test_split(df_negative, test_size = 0.20)
    df_train = pd.concat([df_positive_train,df_negative_train])
    df_val = pd.concat([df_positive_val,df_negative_val])
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)
    Y_train = df_train["label"] 
    Y_val = df_val["label"]
    X_train = df_train.drop(columns=["user_id","register_time","prediction_pay_price","label"])
    X_val = df_val.drop(columns=["user_id","register_time","prediction_pay_price","label"])                
    return X_train, X_val, Y_train, Y_val


# In[159]:


X_train, X_val, Y_train, Y_val = split_train_val(df_positive, df_negative, val_size = 0.2)


# In[165]:


#降低训练集阴性比例 under-sampling
positive_count_train = Y_train.sum()
rus = RandomUnderSampler(ratio={0:positive_count_train*99, 1:positive_count_train}, random_state=0)
X_train_undersampled, y_train_undersampled = rus.fit_sample(X_train, Y_train)
print('y_train_undersample:\n{}'.format(pd.Series(y_train_undersampled).value_counts()))

#提高训练集阳性比例 over-sampling
smote = SMOTE(ratio={0:positive_count_train*99, 1:positive_count_train*10}, random_state=0)
X_train_resampled, y_train_resampled = smote.fit_sample(X_train_undersampled, y_train_undersampled)
print('y_train_resample:\n{}'.format(pd.Series(y_train_resampled).value_counts()))


# In[145]:


classifier = RandomForestClassifier(n_estimators = 150, criterion = 'entropy',class_weight ='balanced')
classifier.fit(X_train_undersampled, y_train_undersampled)


# In[146]:


preds = classifier.predict_proba(X_val)[:,1]


# In[147]:


def optimal_threshold(preds,Y_val):
    for threshold in range(0,10):
        threshold = threshold/10
        print (threshold)
        Y_pred =  np.where(preds<=threshold, 0, 1)
        tp = Y_val[np.logical_and(Y_pred ==1,Y_val==1)].shape[0]
        fp  = Y_val[np.logical_and(Y_pred ==1,Y_val==0)].shape[0]
        tn = Y_val[np.logical_and(Y_pred ==0,Y_val==0)].shape[0]
        fn = Y_val[np.logical_and(Y_pred ==0,Y_val==1)].shape[0]
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 =  2*recall*precision/(recall+precision)
        print(tp,fp,tn,fn)
        print("precision: %g, recall:%g, f1:%g"  % (precision,recall,f1))
        
optimal_threshold(preds,Y_val)


# # Evaluation

# In[148]:


#ROC
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# calculate the fpr and tpr for all thresholds of the classification
probs = classifier.predict_proba(X_val)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_val, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[149]:


# Predicting the Test set results
#Y_pred = classifier.predict(X_val)


# ## Xgboost

# In[150]:


import lightgbm as lgb


# In[166]:


clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=150, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
clf.fit(X_train_undersampled, y_train_undersampled, eval_set=[(X_train_undersampled, y_train_undersampled)], eval_metric='auc',early_stopping_rounds=100)
preds = clf.predict_proba(X_val)[:,1]


# In[167]:


optimal_threshold(preds,Y_val)


# In[168]:


#ROC
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# calculate the fpr and tpr for all thresholds of the classification
probs = clf.predict_proba(X_val)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_val, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

