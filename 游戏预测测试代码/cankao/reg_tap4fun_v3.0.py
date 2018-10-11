# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:22:48 2018
@author: 1707500
"""
'''
V1.0回归  ->85
v2.0 LASSO回归 转化label 为 ([prediction_pay_price]-[pay_price])/[prediction_pay_price] -> 66
v3.0 LASSO回归 转化label 为 ([prediction_pay_price]-[pay_price])/[prediction_pay_price] -> 66

'''
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import lightgbm as lgb
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns       
from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import datetime




# =============================================================================
# import sys
# sys.path.append('..')
# import gluonbook as gb
# from mxnet import autograd, gluon, init, nd
# from mxnet.gluon import loss as gloss, nn
# =============================================================================



'''
basic做数据预处理
'''
def data_process(X):
    X['wood'] = X['wood_add_value'] - X['wood_reduce_value'] 
    X['stone'] = X['stone_add_value'] - X['stone_reduce_value'] 
    X['ivory'] = X['ivory_add_value']  - X['ivory_reduce_value'] 
    X['meat'] =  X['meat_add_value'] - X['meat_reduce_value']
    X['magic'] =  X['magic_add_value'] - X['magic_reduce_value']
    X['infantry'] = X['infantry_add_value'] - X['infantry_reduce_value'] 
    X['cavalry'] = X['cavalry_add_value'] - X['cavalry_reduce_value'] 
    X['shaman'] = X['shaman_add_value'] - X['shaman_reduce_value'] 
    X['wound_infantry'] = X['wound_infantry_add_value'] - X['wound_infantry_reduce_value'] 
    X['wound_cavalry'] =  X['wound_cavalry_add_value'] - X['wound_cavalry_reduce_value']
    X['wound_shaman'] = X['wound_shaman_add_value'] - X['wound_shaman_reduce_value'] 
    X['general_acceleration'] = X['general_acceleration_reduce_value'] - X['general_acceleration_add_value'] 
    X['building_acceleration'] = X['building_acceleration_add_value'] - X['building_acceleration_reduce_value']
    X['reaserch_acceleration'] = X['reaserch_acceleration_add_value'] -  X['reaserch_acceleration_reduce_value']
    X['training_acceleration'] =  X['training_acceleration_add_value'] - X['training_acceleration_reduce_value']
    X['treatment_acceleration'] =  X['treatment_acceleraion_add_value'] - X['treatment_acceleration_reduce_value']  
# =============================================================================
#     X = X.drop(['wood_reduce_value' ,'wood_add_value' ,'stone_add_value' ,'stone_reduce_value' ,'ivory_add_value' ,'ivory_reduce_value' ,'meat_add_value' ,'meat_reduce_value' ,'magic_add_value' ,'magic_reduce_value' ,'infantry_add_value' ,'infantry_reduce_value' ,'cavalry_add_value' ,'cavalry_reduce_value' ,'shaman_add_value' ,'shaman_reduce_value' ,'wound_infantry_add_value' ,'wound_infantry_reduce_value' ,'wound_cavalry_add_value' ,'wound_cavalry_reduce_value' ,'wound_shaman_add_value' ,'wound_shaman_reduce_value' ,'general_acceleration_add_value' ,'general_acceleration_reduce_value' ,'building_acceleration_add_value' ,'building_acceleration_reduce_value' ,'reaserch_acceleration_add_value' ,'reaserch_acceleration_reduce_value' ,'training_acceleration_add_value' ,'training_acceleration_reduce_value' ,'treatment_acceleraion_add_value' ,'treatment_acceleration_reduce_value'],axis=1)
# =============================================================================
    return X


def get_dummies(X):
# =============================================================================
#     X = pd.get_dummies(X, columns = ['register_time','bd_training_hut_level' ,'bd_healing_lodge_level' ,'bd_stronghold_level' ,'bd_outpost_portal_level' ,'bd_barrack_level' ,'bd_healing_spring_level' ,'bd_dolmen_level' ,'bd_guest_cavern_level' ,'bd_warehouse_level' ,'bd_watchtower_level' ,'bd_magic_coin_tree_level' ,'bd_hall_of_war_level' ,'bd_market_level' ,'bd_hero_gacha_level' ,'bd_hero_strengthen_level' ,'bd_hero_pve_level','sr_scout_level' ,'sr_training_speed_level' ,'sr_infantry_tier_2_level' ,'sr_cavalry_tier_2_level' ,'sr_shaman_tier_2_level' ,'sr_infantry_atk_level' ,'sr_cavalry_atk_level' ,'sr_shaman_atk_level' ,'sr_infantry_tier_3_level' ,'sr_cavalry_tier_3_level' ,'sr_shaman_tier_3_level' ,'sr_troop_defense_level' ,'sr_infantry_def_level' ,'sr_cavalry_def_level' ,'sr_shaman_def_level' ,'sr_infantry_hp_level' ,'sr_cavalry_hp_level' ,'sr_shaman_hp_level' ,'sr_infantry_tier_4_level' ,'sr_cavalry_tier_4_level' ,'sr_shaman_tier_4_level' ,'sr_troop_attack_level' ,'sr_construction_speed_level' ,'sr_hide_storage_level' ,'sr_troop_consumption_level' ,'sr_rss_b_prod_level' ,'sr_rss_c_prod_level' ,'sr_rss_d_prod_level' ,'sr_rss_a_gather_level' ,'sr_rss_b_gather_level' ,'sr_rss_c_gather_level' ,'sr_rss_d_gather_level' ,'sr_troop_load_level' ,'sr_rss_e_gather_level' ,'sr_rss_e_prod_level' ,'sr_outpost_durability_level' ,'sr_outpost_tier_2_level' ,'sr_healing_space_level' ,'sr_gathering_hunter_buff_level' ,'sr_healing_speed_level' ,'sr_outpost_tier_3_level' ,'sr_alliance_march_speed_level' ,'sr_pvp_march_speed_level' ,'sr_gathering_march_speed_level' ,'sr_outpost_tier_4_level' ,'sr_guest_troop_capacity_level' ,'sr_march_size_level' ,'sr_rss_help_bonus_level' ])   
# =============================================================================
    X = pd.get_dummies(X, columns = ['register_time'])
    return X

def min_max_scaler(X):    
    min_max_scaler = preprocessing.MinMaxScaler()
    features_new = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(features_new, columns=X.columns)
    return X

tap_fun_train = pd.read_csv("D:/data/tap4fun/tap_fun_train.csv", sep = ',')
tap_fun_test = pd.read_csv("D:/data/tap4fun/tap_fun_test.csv", sep = ',')




'''
数据可视化分析
'''
# =============================================================================
# data_vis = tap_fun_train[tap_fun_train.prediction_pay_price > 0]
# corrmat = data_vis.corr()
# f, ax = plt.subplots(figsize=(40, 40))
# sns.heatmap(corrmat, vmax=0.8, square=True)
# 
# 
# 
# k  = 20 # 关系矩阵中将显示10个特征
# cols = corrmat.nlargest(k, 'prediction_pay_price')['prediction_pay_price'].index
# cm = np.corrcoef(data_vis[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, \
#                  square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
# 
# 
# =============================================================================



tap_fun_train['xiangcha'] = tap_fun_train['prediction_pay_price']-tap_fun_train['pay_price']
tap_fun_train['label'] = tap_fun_train['prediction_pay_price']
tap_fun_train = tap_fun_train.fillna(0)

tap_fun_train['classification_label'] = tap_fun_train['xiangcha'].map(lambda x: 1 if x > 0 else 0)
data = pd.concat([tap_fun_train,tap_fun_test])
data = data.fillna(-1)
register_time = []
for i in data['register_time']:
# =============================================================================
#     print(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').strftime("%a"))
# =============================================================================
    register_time.append(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').strftime("%a"))
data['register_time'] = register_time

print(data['classification_label'].value_counts())

'''
分类数据预处理
'''
data = data_process(data)
# =============================================================================
# data = get_dummies(data)
# =============================================================================
print('dummies finish!!')


'''
回归建模测试集数据准备
'''
reg_data = data_process(data)
# =============================================================================
# reg_data = get_dummies(reg_data)
# =============================================================================
reg_test = data[data.classification_label == -1].drop(['user_id','register_time','classification_label','prediction_pay_price','xiangcha','label'],axis=1)
# =============================================================================
# reg_test = data[data.classification_label == -1][['pay_price','ivory_add_value','stone_add_value','wood_add_value','ivory_reduce_value']]
# =============================================================================
# =============================================================================
# reg_test = min_max_scaler(reg_test)
# =============================================================================
reg_test['intercept'] = 1.0

'''
训练回归模型
'''
from lightgbm import LGBMRegressor

reg_data = reg_data[(reg_data.label >= 0)]
# =============================================================================
# reg_data = reg_data.drop((reg_data[(reg_data.pay_price >= 100) & (reg_data.avg_online_minutes < 35)]).index)
# reg_data = reg_data.drop((reg_data[(reg_data.pay_price < 0.99) & (reg_data.avg_online_minutes > 840)]).index)
# =============================================================================

reg_target = reg_data['label']
reg_features = reg_data.drop(['user_id','register_time','classification_label','prediction_pay_price','xiangcha','label'],axis=1)
# =============================================================================
# reg_features = reg_data[['pay_price','ivory_add_value','stone_add_value','wood_add_value','ivory_reduce_value']]
# =============================================================================
# =============================================================================
# reg_features = min_max_scaler(reg_features)
# =============================================================================
reg_features['intercept'] = 1.0

'''
'''
# =============================================================================
# from sklearn import linear_model
# reg = linear_model.Lasso(alpha=0.1, fit_intercept=True, normalize=True, precompute=True, copy_X=True, max_iter=1000, tol=0.0001, warm_start=True, positive=True, random_state=42, selection='cyclic')
# 
# =============================================================================

from sklearn.linear_model import ElasticNet
reg = ElasticNet(alpha=0.8, l1_ratio=0.8, fit_intercept=True, normalize=False, precompute=True, copy_X=True, max_iter=1000, tol=0.001, warm_start=True, positive=True, random_state=42, selection='cyclic')


# =============================================================================
# reg = LGBMRegressor(num_leaves=40,max_depth=7,n_estimators=3000,min_child_weight=10,subsample=0.7, 
#                     colsample_bytree=0.7,reg_alpha=0,learning_rate=0.1,reg_lambda=1,bagging_fraction = 0.8,
#                     bagging_freq = 5,feature_fraction_seed=9, bagging_seed=9)
# 
# =============================================================================



'''
实际回归建模
'''

cnt=1
size = math.ceil(len(reg_features) / cnt)
result=[]
print('ready for reg!!')
for i in range(cnt):
    start = size * i
    end = (i + 1) * size if (i + 1) * size < len(reg_features) else len(reg_features)
    slice_features = reg_features[start:end]
    slice_target = reg_target[start:end]
    print(i+1)
    X_train,X_test,y_train,y_test = train_test_split(slice_features,slice_target,test_size=0.2,random_state=42)
    reg.fit(X_train, y_train) 

# =============================================================================
#     reg.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test, y_test)],eval_names =['train_data','valid_data'], eval_metric='rmse',early_stopping_rounds=100) 
# =============================================================================
    
    y_pre = reg.predict(X_test)
    print(np.sqrt(metrics.mean_squared_error(y_test, y_pre)))
    y_pred = reg.predict(reg_test)
    result.append(y_pred)
    
y_pred = np.mean(result,axis=0)


reg_df = pd.DataFrame({ 
                        'user_id' : data[data.classification_label == -1]['user_id'],
                        'price':reg_test['pay_price'],
                        'prediction_pay_price' : y_pred
                        })
 
reg_df_1 = reg_df[reg_df.price == 0]
reg_df_2 = reg_df[reg_df.price != 0]

reg_df_1['prediction_pay_price'] = reg_df_1['prediction_pay_price'].map(lambda x: 0 if x < 30 else x)
reg_df = pd.concat([reg_df_1,reg_df_2])

reg_df[['user_id','prediction_pay_price']].to_csv('D:/999github/kaggle/sub_sample.csv', index=False)

reg_df.to_csv('D:/999github/kaggle/sub_sample_10.csv', index=False)

'''
回归特征权重显示
'''

importances = reg.coef_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(reg_features.shape[1]):
    print("%d. feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]] , reg_features.columns[indices[f]] ))

