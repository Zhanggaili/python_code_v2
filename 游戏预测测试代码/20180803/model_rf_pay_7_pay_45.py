# -*- coding: utf-8 -*-
# @Date     :2018/7/31 21:25
# @Author   :cq_yang
# @:Describe:

# 导入必要的工具包
import numpy as np  # 用于数值计算
import pandas as pd     # 用于数据表处理，数据文件读写
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.linear_model import SGDRegressor
import warnings
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
warnings.filterwarnings("ignore")

from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt     # 用于绘图：分析结果的可视化。

pd.set_option('display.float_format', lambda x: '%.4f' % x)

comman_path = r"../"
Y_Standard = False
train_path = comman_path + r"data/original/train_pay_7_pay_45.csv"

test_path = comman_path + r"data/original/test_pay_7_pay_45.csv"

save_path = comman_path + r"data/submit/result_pay_7_pay_45_xgb.csv"
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
import gc


class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self, additional=1):
        self.additional = additional

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 时间特征


        # 战斗特征
        X["pve_vs_price"] = X["pve_win_count"] / (X["pay_price"] + 0.1)
        X["pvp_vs_price"] = X["pvp_win_count"] / (X["pay_price"] + 0.1)


        # 材料特征
        X['wood2'] = -X['wood_add_value'] - X['wood_reduce_value']
        X['wood'] = X['wood_add_value'] - X['wood_reduce_value']
        X['wood1'] = X['wood']/(0.1+X['wood_add_value'])
        X['stone2'] = -X['stone_add_value'] - X['stone_reduce_value']
        X['stone'] = X['stone_add_value'] - X['stone_reduce_value']
        X['stone1'] = X['stone'] / (0.1 + X['stone_add_value'])
        X['ivory2'] = -X['ivory_add_value'] - X['ivory_reduce_value']
        X['ivory'] = X['ivory_add_value'] - X['ivory_reduce_value']
        X['ivory1'] = X['ivory'] / (0.1 + X['ivory_add_value'])
        X["ivory3"] = (X["ivory_reduce_value"]+0.1)/(X["ivory_add_value"]+0.1)

        X['meat'] = -X['meat_add_value'] + X['meat_reduce_value']

        X["wood3"] = (X["wood2"])*(X["pay_price"]/X["pay_count"])
        X["stone3"] = (X["stone2"]) * (X["pay_price"] / X["pay_count"])
        X["ivory4"] = (X["ivory2"]) * (X["pay_price"] / X["pay_count"])
        X["meat3"] = (X["meat"]) * (X["pay_price"] / X["pay_count"])

        # X['meat1'] = X['meat'] / (0.01 + X['meat_add_value'])
        # X['magic'] = -X['magic_add_value'] - X['magic_reduce_value']
        # X['magic'] = X['magic'] / (0.1 + X['magic_add_value'])
        # X['infantry2'] = -X['infantry_add_value'] - (X['infantry_reduce_value'])
        X['infantry'] = X['infantry_add_value']-(X['infantry_reduce_value'])
        X['infantry'] = X['infantry'] / (0.1 + X['infantry_add_value'])
        X["infantry3"] = (X["infantry"]) * (X["pay_price"] / X["pay_count"])
        # X['cavalry'] = -X['cavalry_add_value'] + X['cavalry_reduce_value']
        X['shaman'] = X['shaman_add_value'] - X['shaman_reduce_value']
        X['shaman'] = X['shaman'] / (0.1 + X['shaman_add_value'])
        # X['wood'] = (X['wound_infantry_add_value'] - X['wound_infantry_reduce_value'])

        X['general_acceleration'] = X['general_acceleration_reduce_value'] - X['general_acceleration_add_value']
        X['building_acceleration'] = X['building_acceleration_add_value'] - X['building_acceleration_reduce_value']
        X['reaserch_acceleration'] = X['reaserch_acceleration_reduce_value'] + X['reaserch_acceleration_add_value']
        X['training_acceleration'] = X['training_acceleration_add_value'] - X['training_acceleration_reduce_value']
        X['treatment_acceleration'] = X['treatment_acceleraion_add_value'] - X['treatment_acceleration_reduce_value']
        X["treatment_acceleration3"] = (X["treatment_acceleration"]) * (X["pay_price"] / X["pay_count"])
        X['training_acceleration_speed'] = X['training_acceleration'] * X["sr_training_speed_level"]

        X['building_acceleration_per'] = (X['building_acceleration'] * X["bd_training_hut_level"]).apply(lambda x:abs(x))
        X['treatment_acceleration1'] = (X['treatment_acceleration'] * X["bd_healing_lodge_level"])

        X['treatment_acceleration_infantry'] = X['treatment_acceleration'] * X["wound_infantry_reduce_value"]
        # X['treatment_acceleration_cavalry'] = X['treatment_acceleration'] * X["wound_cavalry_reduce_value"]
        # X['treatment_acceleration_consumption'] = X['treatment_acceleration'] * X["sr_troop_consumption_level"]
        # X['treatment_acceleration_speed'] = X['treatment_acceleration'] * X["sr_healing_speed_level"]


        X['bd_healing_spring_infantry'] = X['wound_infantry_reduce_value'] /( X["bd_healing_spring_level"]+0.1)
        X['bd_healing_spring_cavalry'] = X['wound_cavalry_reduce_value'] /( X["bd_healing_spring_level"]+0.1)




        X["outpost_pay"] = X["sr_outpost_durability_level"] * X["pay_price"]
        X["bd_training_hut_pay"] = X["bd_training_hut_level"] * X["pay_price"]
        X["time_pay"] = X["avg_online_minutes"] * X["pay_price"] # xianshangxianxiabuyizhi


        X['infantry'] = X['infantry_add_value'] - X['infantry_reduce_value']
        X['cavalry'] = X['cavalry_add_value'] - X['cavalry_reduce_value']
        X['wound_shaman'] = X['wound_shaman_add_value'] - X['wound_shaman_reduce_value']

        X["building_level"] = X.loc[:, X.columns.str.match('bd_.+_level')].sum(axis=1)




        dt1 = pd.to_datetime(X["register_time"])
        X["register_time"] = dt1
        X["wkd"] = dt1.dt.weekday
        X["day"] = dt1.dt.day
        X["hour"] = dt1.dt.hour
        X["day_in_year"] = dt1.dt.dayofyear
        X["days_spfest"] = X["register_time"].map(lambda x: (x - pd.to_datetime("2018-02-16")).days)



        bd_cols_lv = [col for col in X.columns if col.startswith("bd")]
        sr_cols_lv = [col for col in X.columns if col.startswith("sr")]
        X["total_bd_lv"] = X[bd_cols_lv].sum(axis=1)
        X["total_sr_lv"] = X[sr_cols_lv].sum(axis=1)

        X["total_lv"] = X["total_bd_lv"] + X["total_sr_lv"]



        # X["total_bd_lv_vs_price"] = X["total_sr_lv"]/(X["pay_price"]+0.1)
        # X["pve_vs_price"] = X["pve_win_count"] / (X["pay_price"] + 0.1)
        # X["pvp_vs_price"] = X["pvp_win_count"] / (X["pay_price"] + 0.1)

        new_mean = pd.DataFrame(X.mean()).T

        new_train = X[new_mean.columns]
        all_mean = pd.DataFrame(new_train.values - new_mean.values, columns=new_mean.columns, index=new_train.index)
        all_mean = (all_mean >= 0).astype(int).sum(axis=1)
        # X["all_mean"] = all_mean*1






        X = X.drop(['wood_reduce_value', 'wood_add_value', 'stone_add_value', 'stone_reduce_value', 'ivory_add_value',
                    'ivory_reduce_value', 'meat_add_value', 'meat_reduce_value', 'magic_add_value',
                    'magic_reduce_value', 'infantry_add_value', 'infantry_reduce_value', 'cavalry_add_value',
                    'cavalry_reduce_value', 'shaman_add_value', 'shaman_reduce_value', 'wound_infantry_add_value',
                    'wound_infantry_reduce_value', 'wound_cavalry_add_value', 'wound_cavalry_reduce_value',
                    'wound_shaman_add_value', 'wound_shaman_reduce_value', 'general_acceleration_add_value',
                    'general_acceleration_reduce_value', 'building_acceleration_add_value',
                    'building_acceleration_reduce_value', 'reaserch_acceleration_add_value',
                    'reaserch_acceleration_reduce_value', 'training_acceleration_add_value',
                    'training_acceleration_reduce_value', 'treatment_acceleraion_add_value',
                    'treatment_acceleration_reduce_value'], axis=1)

        gc.collect()
        return X


def compare(train_y,prediction):
    #     for i in range(len(train_y)):
    #         print(train_y[i],prediction[i])
    print('The rmse of prediction is:', mean_squared_error(train_y, prediction) ** 0.5)
    print('The R2 of prediction is:',r2_score(train_y, prediction))
    #   观察预测值与真值的散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(train_y, prediction)
    plt.plot([-3, 3], [-3, 3], '--k')   # 数据已经标准化，3倍标准差即可
    plt.axis('tight')
    plt.xlabel('True price')
    plt.ylabel('Predicted price')
    plt.tight_layout()
    # plt.show()


def model(train_path,test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    pipe = Pipeline([
        ('add_feature', add_feature(additional=2))
    ])
    train_df = pipe.fit_transform(train_df)
    test_df = pipe.fit_transform(test_df)
    ss_X = StandardScaler()
    ss_y = StandardScaler()

    cols = [x for x in test_df.columns if x not in ["register_time", "user_id", "prediction_pay_price", "leibie"]]
    train_X = ss_X.fit_transform(train_df[cols])
    test_X = ss_X.fit(test_df[cols])

    # train_y = train_df["prediction_pay_price"].values
    train_y = ss_y.fit_transform(train_df["prediction_pay_price"].values.reshape(-1, 1))

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'square loss',
        'learning_rate': 0.003,
        'num_leaves': 31,
        'max_bin': 255,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0,
        'lambda_l2': 0,
        'min_split_gain': 0
    }





    from sklearn.linear_model import ElasticNet
    reg = ElasticNet(alpha=10000, l1_ratio=0.8, fit_intercept=True, normalize=False, precompute=True, copy_X=True,
                     max_iter=2000, tol=0.001, warm_start=True, positive=True, random_state=2018, selection='cyclic')
    reg_features = train_X
    reg_features = train_df[cols]
    from sklearn import metrics
    import math
    # reg_target =train_y
    reg_target = train_df['prediction_pay_price']
    cnt = 1
    size = math.ceil(len(reg_features) / cnt)
    result = []

    print('ready for reg!!')
    for i in range(cnt):
        start = size * i
        end = (i + 1) * size if (i + 1) * size < len(reg_features) else len(reg_features)
        slice_features = reg_features[start:end]
        slice_target = reg_target[start:end]
        print(i + 1)
        X_train, X_test, y_train, y_test = train_test_split(slice_features, slice_target, test_size=0.2,
                                                            random_state=42)
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
        # reg = lgb.train(params, lgb_train, num_boost_round=3000)

        # y_pre = reg.predict(X_test, num_iteration=reg.best_iteration)





        reg.fit(X_train, y_train)

        y_pre = reg.predict(X_test)
        # print(np.sqrt(metrics.mean_squared_error(y_test, y_pre)))
        # y_test=ss_y.inverse_transform(y_test)
        # y_pre=ss_y.inverse_transform(y_pre)
        compare(y_test, y_pre)
        # y_pred = reg.predict(test_X)
        y_pred = reg.predict(test_df[cols])
        result.append(y_pred)
    y_pred = np.mean(result, axis=0)
    # importances = reg.coef_
    # indices = np.argsort(importances)[::-1]
    # print("Feature ranking:")
    # for f in range(reg_features.shape[1]):
    #     print("%d. feature %d (%f): %s" % (
    #     f + 1, indices[f], importances[indices[f]], reg_features.columns[indices[f]]))
    return y_pred


def generate_summit(predict,test_path,save_path):
    test_df = pd.read_csv(test_path)
    testpredict = test_df.copy()
    testpredict["prediction_pay_price"] = predict
    testpredict = testpredict[["user_id", "prediction_pay_price"]]
    testpredict["prediction_pay_price"] = testpredict["prediction_pay_price"].apply(lambda x: x if x > 0 else 0)
    testpredict = testpredict.sort_values('prediction_pay_price', ascending=False)
    testpredict.to_csv(save_path,index=0)


if __name__ == "__main__":
    predict = model(train_path, test_path)
    generate_summit(predict, test_path, save_path)
