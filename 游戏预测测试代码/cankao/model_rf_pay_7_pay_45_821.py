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
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
warnings.filterwarnings("ignore")

from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt     # 用于绘图：分析结果的可视化。

pd.set_option('display.float_format', lambda x: '%.4f' % x)

comman_path = r"../"
Y_Standard = False
train_path = comman_path + r"data/original/train_pay_7_pay_45.csv"

test_path = comman_path + r"data/original/test_pay_7_pay_45_st.csv"

save_path = comman_path + r"data/submit/result_pay_7_pay_45_st.csv"
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
        # X["pve_vs_price"] = X["pve_win_count"] * (X["pay_price"] + 0.1)
        # X["pvp_vs_price"] = X["pvp_win_count"] * (X["pay_price"] + 0.1)
        X["pve_vs_price"] = (X["pve_lanch_count"]-X["pve_win_count"])*X["pay_price"]
        X["pvp_loss_price"] = (X["pvp_battle_count"]-X["pvp_win_count"]+X["pve_battle_count"]-X["pve_win_count"])*(X["pay_count"])


        X["pvp_beidong_pve"] = (X["pvp_lanch_count"]-X["pvp_battle_count"])*(X["pve_lanch_count"]-X["pve_battle_count"])
        X["pvp_win_pve"] = X["pvp_win_count"] *X["pve_win_count"]
        X["pvp_lanch_pve"] = X["pvp_lanch_count"] * X["pve_lanch_count"]
        X["pvp_battle_pve"] = X["pvp_battle_count"] * X["pve_battle_count"]
        X["pvp_battle_pve"] = X["pvp_battle_count"] * X["pve_battle_count"]
        X["pvp_win_rate"] =  (X["pvp_win_count"]) /(X["pvp_battle_count"]+0.01)
        X["pve_win_rate"] = (X["pve_win_count"] ) / (X["pve_battle_count"] + 0.01)
        X["pvp_lanch_rate"] = (X["pvp_lanch_count"]) /(X["pvp_battle_count"]+0.01)
        X["pve_lanch_rate"] = (X["pve_lanch_count"] ) / (X["pve_battle_count"] + 0.01)
        X["pvp_rate_pve"] = X["pvp_win_rate"] * X["pve_win_rate"]



        # 材料特征
        X['wood2'] = -X['wood_add_value'] - X['wood_reduce_value']
        X['wood'] = X['wood_add_value'] - X['wood_reduce_value']
        X['wood1'] = X['wood']/(0.01+X['wood_add_value'])
        X['stone2'] = -X['stone_add_value'] - X['stone_reduce_value']
        X['stone'] = X['stone_add_value'] - X['stone_reduce_value']
        X['stone1'] = X['stone'] / (0.01 + X['stone_add_value'])
        X['ivory2'] = -X['ivory_add_value'] - X['ivory_reduce_value']
        X['ivory'] = X['ivory_add_value'] - X['ivory_reduce_value']
        X['ivory1'] = X['ivory'] / (0.01 + X['ivory_add_value'])
        X["ivory3"] = (X["ivory_reduce_value"])/(X["ivory_add_value"]+0.01)

        X['meat'] = -X['meat_add_value'] + X['meat_reduce_value']

        X["wood3"] = (X["wood2"])*(X["pay_price"]/(X["pay_count"]))
        X["stone3"] = (X["stone2"]) * (X["pay_price"] / (X["pay_count"]))
        X["ivory4"] = (X["ivory2"]) * (X["pay_price"] / (X["pay_count"]))
        X["meat3"] = (X["meat"]) * (X["pay_price"] / (X["pay_count"]))

        # X['meat1'] = X['meat'] / (0.01 + X['meat_add_value'])
        X['magic'] = -X['magic_add_value'] - X['magic_reduce_value']






        #########
        # X['magic'] = X['magic'] / (0.1 + X['magic_add_value'])
        X['infantry2'] = -X['infantry_add_value'] + X['infantry_reduce_value']
        X['infantry'] = X['infantry_add_value']-X['infantry_reduce_value']
        X['infantry1'] = X['infantry'] / (0.01 + X['infantry_add_value'])
        X["infantry3"] = (X["infantry"]) * (X["pay_price"] / (X["pay_count"]))
        # X['cavalry'] = -X['cavalry_add_value'] - X['cavalry_reduce_value']
        X['shaman'] = X['shaman_add_value'] - X['shaman_reduce_value']
        X['shaman'] = X['shaman'] / (0.01 + X['shaman_add_value'])

        X['general_acceleration'] = X['general_acceleration_reduce_value'] - X['general_acceleration_add_value']
        X['building_acceleration'] = X['building_acceleration_add_value'] - X['building_acceleration_reduce_value']
        X['reaserch_acceleration'] = X['reaserch_acceleration_reduce_value'] + X['reaserch_acceleration_add_value']
        X['training_acceleration'] = X['training_acceleration_add_value'] - X['training_acceleration_reduce_value']
        X['treatment_acceleration'] = X['treatment_acceleraion_add_value'] - X['treatment_acceleration_reduce_value']

        # X["treatment_acceleration2"] = -(X["bd_healing_lodge_level"]+X["bd_healing_spring_level"])  * (X["pay_price"] / X["pay_count"])

        X["treatment_acceleration3"] = (X["treatment_acceleration"]) * (X["pay_price"] / (X["pay_count"]))
        X['training_acceleration_speed'] = X['training_acceleration'] * X["sr_training_speed_level"]

        X['building_acceleration_per'] = (X['building_acceleration'] * X["bd_training_hut_level"]).apply(lambda x:abs(x))
        X['treatment_acceleration1'] = (X['treatment_acceleration'] * X["bd_healing_lodge_level"])

        X['treatment_acceleration_infantry'] = X['treatment_acceleration'] * X["wound_infantry_reduce_value"]


        X['bd_healing_spring_infantry'] = X['wound_infantry_reduce_value'] /( X["bd_healing_spring_level"]+0.01)
        X['bd_healing_spring_cavalry'] = X['wound_cavalry_reduce_value'] /( X["bd_healing_spring_level"]+0.01)


        # X["wound_1"] = (X["wound_infantry_add_value"]+X["wound_cavalry_add_value"]+X["wound_shaman_add_value"]
        #              -(X["wound_infantry_reduce_value"]+X["wound_cavalry_reduce_value"]+X["wound_shaman_reduce_value"]))/\
        #             (0.1+X["pvp_battle_count"]+X["pve_battle_count"]-(X["pvp_battle_count"]+X["pve_battle_count"]))*\
        #                X["treatment_acceleraion_add_value"]
        #
        # X["wound_2"] = (X["wound_infantry_add_value"] + X["wound_cavalry_add_value"] + X["wound_shaman_add_value"]
        #                 - (X["wound_infantry_reduce_value"] + X["wound_cavalry_reduce_value"] + X[
        #             "wound_shaman_reduce_value"])) / \
        #                (0.1 + X["pvp_battle_count"] + X["pve_battle_count"] - (
        #                        X["pvp_battle_count"] + X["pve_battle_count"])) * X["treatment_acceleration"]

        X["wound_3"] = ((X["wound_infantry_reduce_value"] + X["wound_cavalry_reduce_value"] + X[
                        "wound_shaman_reduce_value"])) / (X["treatment_acceleration"]+0.01)
        # X["wound_4"] = ((X["wound_infantry_add_value"] + X["wound_cavalry_add_value"] + X[
            # "wound_shaman_add_value"])) / (X["pvp_battle_count"] + 0.1)


        X["stronghold_vs_price"] =  X["bd_stronghold_level"]*X["pay_price"]
        # X["stronghold_vs_price2"] = X["stronghold_vs_price"]/X["pay_count"]
        X["outpost_vs_price"] = X["bd_outpost_portal_level"] * X["pay_price"]
        X["market_vs_price"] = X["bd_market_level"]* X["pay_price"]
        X["troop_vs_price"] = X["sr_troop_load_level"] * X["pay_price"]
        X["ivory_vs_count"] = X["ivory_add_value"] /(X["pay_count"])
        X["ivory_use_rate"] = (X["ivory_reduce_value"] ) / (X["ivory_add_value"] + 0.01)
        X["acc_count"] = (X["general_acceleration_add_value"]+X["building_acceleration_add_value"]\
                         + X["reaserch_acceleration_add_value"]+X["training_acceleration_add_value"]\
                         + X["treatment_acceleraion_add_value"])\
                         /(-X["pve_lanch_count"]+0.01)

        # X["ivory_pvp"] = X["ivory_add_value"]/(-X["pvp_battle_count"]-X["pvp_win_count"]+0.1)
        X["stone_pvp"] = X["stone_reduce_value"] / (-X["pvp_battle_count"] - X["pvp_win_count"] + 0.01)

        X["meat_pvp"] = X["meat_reduce_value"] / (-X["pvp_battle_count"] - X["pvp_win_count"] + 0.01)
        X["wood_pvp"] = X["wood_reduce_value"] / (-X["pvp_battle_count"] - X["pvp_win_count"] + 0.01)



        X["pve_lanch_time"] = X["pve_lanch_count"]*X["avg_online_minutes"]


        X["pvp_pay_count"] = X["pay_count"] * (X["pvp_lanch_count"]+X["pvp_win_count"])
        X["pve_pay_count"] = X["pay_count"] * (X["pve_lanch_count"] + X["pve_win_count"])





        # X["wound_infantry"] = (X["wound_infantry_add_value"]-X["wound_infantry_reduce_value"])+X["infantry_reduce_value"]



        X["outpost_pay"] = X["sr_outpost_durability_level"] * X["pay_price"]
        X["bd_training_hut_pay"] = X["bd_training_hut_level"] * X["pay_price"]
        # X["time_pay"] = X["avg_online_minutes"] * X["pay_price"] # xianshangxianxiabuyizhi
        X["time_pay_1"] = X["avg_online_minutes"] * X["pay_price"]/(X["pay_count"])


        X['infantry'] = X['infantry_add_value'] - X['infantry_reduce_value']
        X['cavalry'] = X['cavalry_add_value'] - X['cavalry_reduce_value']
        X['wound_shaman'] = X['wound_shaman_add_value'] - X['wound_shaman_reduce_value']

        X["building_level"] = X.loc[:, X.columns.str.match('bd_.+_level')].sum(axis=1)

        X["bing"] = (-X["shaman_add_value"]-X["cavalry_add_value"]+X["infantry_add_value"]) *X["pvp_win_count"]/(X["pvp_battle_count"]+0.01)



        X["infantry_all"] =-(X["sr_infantry_atk_level"]+X["sr_infantry_def_level"]+X["sr_infantry_hp_level"]) * X["pay_price"]



        X["atk_level"] = (X["sr_shaman_atk_level"]+X["sr_cavalry_atk_level"]+X["sr_infantry_atk_level"])*X["pay_price"]
        X["hp_level"] = (X["sr_shaman_hp_level"] + X["sr_cavalry_hp_level"] + X["sr_infantry_hp_level"]) * X[
            "pay_price"]/(X["pay_count"])




        dt1 = pd.to_datetime(X["register_time"])
        X["register_time"] = dt1
        X["wkd"] = dt1.dt.weekday
        X["day"] = dt1.dt.day
        X["hour"] = dt1.dt.hour
        X["day_in_year"] = dt1.dt.dayofyear
        X["days_spfest"] = X["register_time"].map(lambda x: (x - pd.to_datetime("2018-02-16")).days)
        # X["hour_1"] = X.hour.apply(lambda x:1 if x in [2,3,4,13,14,15] else 0)
        # X["days_yuanxiao"] = X["register_time"].map(lambda x: (x - pd.to_datetime("2018-03-02")).days)


        bd_cols_lv = [col for col in X.columns if col.startswith("bd")]
        sr_cols_lv = [col for col in X.columns if col.startswith("sr")]
        X["total_bd_lv"] = X[bd_cols_lv].sum(axis=1)
        X["total_sr_lv"] = X[sr_cols_lv].sum(axis=1)

        X["total_lv"] = X["total_bd_lv"] + X["total_sr_lv"]


        X["total_bd_lv_vs_price"] = (X["total_bd_lv"])*(X["pay_price"])
        X["pve_vs_price"] = X["pve_win_count"] / (X["pay_price"])
        X["pvp_vs_price"] = X["pvp_win_count"] / (X["pay_price"])


        new_mean = pd.DataFrame(X.mean()).T

        new_train = X[new_mean.columns]
        all_mean = pd.DataFrame(new_train.values - new_mean.values, columns=new_mean.columns, index=new_train.index)
        all_mean = (all_mean >= 0).astype(int).sum(axis=1)
        X["all_mean"] = all_mean*1

        X["zhangli_count"] = ((X["infantry_add_value"] + X["cavalry_add_value"] + X[
            "shaman_reduce_value"]) - (X["infantry_reduce_value"] + X["cavalry_add_value"] + X[
            "shaman_reduce_value"]))/(X["pay_count"])
        X["zhangli_count_price"] = ((X["infantry_add_value"] + X["cavalry_add_value"] + X[
            "shaman_reduce_value"]) - (X["infantry_reduce_value"] + X["cavalry_add_value"] + X[
            "shaman_reduce_value"])) / (X["pay_price"])*X["pay_count"]

        # X["pvp_bing_1"]= (X["pvp_battle_count"])\
        #                *(0.01+X["infantry_add_value"]+X["cavalry_add_value"]+X["shaman_add_value"])

        X["pujie"] = (-X['pvp_battle_count']-X["pvp_win_count"])*X["pay_price"]/X["pay_count"]

        # X["hour_new"] = X.hour.apply(lambda x:5 if x in [2, 3, 4, 13, 14, 15] else 2)
        # X["wkd_new"] = X.hour.apply(lambda x: 5 if x in [1, 5, 6, 7] else 2)
        # X["hour_new_1"] = X["hour_new"] * X["pay_price"]
        # X["wkd_new_2"] = (X["wkd_new"] * X["pay_price"] + X["hour_new"] * X["pay_price"])
        # X["wkd_new_1"] = X["wkd_new"] * X["pay_count"]
##################


        X["hour_1"] = X["hour"]*X["pay_price"]
        X["wkd_2"] = (X["wkd"] * X["pay_price"]+X["hour"]*X["pay_price"])*0.3
        X["wkd_1"] = X["wkd"] * X["pay_count"]
        # X["day_in_year_1"]= X["day_in_year"] * X["pay_price"]/X["pay_count"]   #zheli1youhemeiyouhendaquebie
        X["day_in_year_1"] = X["day_in_year"] * X["pay_price"]*0.23
        # X["all_feng1"] = (X['wood_add_value'] - 158249380
        #                   + X['stone_add_value'] - 132879785
        #                   + X['ivory_add_value'] - 127705203
        #                   + X['magic_add_value'] - 263722820)/X["avg_online_minutes"]
        # X["all_feng2"] = (X['wood_reduce_value'] - 158249380
        #                   + X['stone_reduce_value'] - 132879785
        #                   + X['ivory_reduce_value'] - 127705203
        #                   + X['magic_reduce_value'] - 263722820)/X["avg_online_minutes"]
        # X["stone_feng"] = (X['stone_add_value'] - 132879785) / X["pay_price"]
        # X["ivory_feng"] = (X['ivory_add_value'] - 127705203) * X["pay_count"]


        X["shaman_all"] = (-X["sr_shaman_atk_level"] - X["sr_shaman_def_level"] - X["sr_shaman_hp_level"]) * X[
            "pay_price"]
        # X["cavalry_all"] = (X["sr_cavalry_atk_level"] -X["sr_cavalry_def_level"] - X["sr_cavalry_hp_level"]) * X[
        #     "pay_price"]
        # X["infantry_all"] = -(X["sr_infantry_atk_level"] +X["sr_infantry_def_level"] + X["sr_infantry_hp_level"]) * X[
        #     "pay_price"]
        X["pujie_2"] = (X["pvp_lanch_count"]+X["pvp_win_count"]) * X["pay_price"]
        X["pujie_3"] = (X["pve_lanch_count"] + X["pve_win_count"]) * X["pay_price"]
        X["pujie_4"] = (-X["pvp_battle_count"] + X["pve_battle_count"]) * X["pay_price"]
        X["pujie_5"] = (-X["pvp_battle_count"] + X["pvp_win_count"]) * X["pay_price"]
        # X["pujie_4"] = (X["pve_lanch_count"] - X["pve_win_count"]) * X["pay_price"]




        # X["pvp_add_pve_consumption"] = (X['pvp_lanch_count'] + X["pve_lanch_count"])*X["sr_troop_consumption_level"]
        # X["bing_consumption"] = (X['infantry_add_value'] + X["cavalry_add_value"]+
        #                          X["shaman_add_value"]-(X['infantry_reduce_value'] +
        #                                                 X["cavalry_reduce_value"]+
        #                                                 X["shaman_reduce_value"]))\
        #                         *X["sr_troop_consumption_level"]


        # X["treatment_acceleraion_level"] = X["pvp_battle_count"]*X["treatment_acceleration_reduce_value"]

        # X["gongji"] = X["sr_troop_attack_level"] * (X["pvp_lanch_count"])
        X = X.drop(X[X.user_id == 1256617].index)


        X["wood_gather"] = X["wood_add_value"] * X["sr_rss_a_gather_level"]
        # X["stone_gather"] = X["stone_add_value"] / (X["sr_rss_b_gather_level"]+0.01)
        # X["ivory_gather"] = X["ivory_add_value"] * X["sr_rss_c_gather_level"]
        # X["meat_gather"] = X["meat_add_value"] - X["sr_rss_d_gather_level"]*X["sr_rss_d_gather_level"]




        X = X.drop([
                    'wood_reduce_value',
                    'wood_add_value',
                    'stone_add_value',
                    'stone_reduce_value',
                    'ivory_add_value',
                    'ivory_reduce_value',
                    'meat_add_value',
                    'meat_reduce_value',
                    'magic_add_value',
                    'magic_reduce_value',
                    'infantry_add_value',
                    'infantry_reduce_value',
                    'cavalry_add_value',
                    'cavalry_reduce_value',
                    'shaman_add_value',
                    'shaman_reduce_value',
                    'wound_infantry_add_value',
                    'wound_infantry_reduce_value',
                    'wound_cavalry_add_value',
                    'wound_cavalry_reduce_value',
                    'wound_shaman_add_value',
                    'wound_shaman_reduce_value',
                    'general_acceleration_add_value',
                    'general_acceleration_reduce_value',
                    'building_acceleration_add_value',
                    'building_acceleration_reduce_value',
                    'reaserch_acceleration_add_value',
                    'reaserch_acceleration_reduce_value',
                    'training_acceleration_add_value',
                    'training_acceleration_reduce_value',
                    'treatment_acceleraion_add_value',
                    'treatment_acceleration_reduce_value',
                    ], axis=1)

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

    cols = [x for x in test_df.columns if x not in ["register_time", "user_id", "prediction_pay_price", "zhongshu","leibie"]]
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
    reg = ElasticNet(alpha=8400, l1_ratio=0.8, fit_intercept=True, normalize=False, precompute=True, copy_X=True,
                     max_iter=2000, tol=0.001, warm_start=True, positive=True, random_state=2018, selection='cyclic')
    # reg2 = RandomForestRegressor()
    # reg3 = ExtraTreesRegressor()
    # reg4 = GradientBoostingRegressor()

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

        # cols = [x for x in train_df.columns if x not in ["register_time", "user_id",  "leibie"]]
        # train_df = train_df[cols]
        # train = train_df[(train_df.day_in_year <= 31)]
        # val = train_df[(train_df.day_in_year > 31)]
        # col = [x for x in train.columns if x not in ["prediction_pay_price"]]
        # X_train = train[col]
        # X_test = val[col]
        # y_train = train["prediction_pay_price"]
        # y_test = val["prediction_pay_price"]

        # cols = [x for x in test_df.columns if x not in ["register_time", "user_id", "leibie","zhongshu"]]
        # lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        # lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
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


        # ##################
        # reg2.fit(X_train, y_train)
        #
        # y_pre = reg2.predict(X_test)
        # # print(np.sqrt(metrics.mean_squared_error(y_test, y_pre)))
        # # y_test=ss_y.inverse_transform(y_test)
        # # y_pre=ss_y.inverse_transform(y_pre)
        # compare(y_test, y_pre)
        # # y_pred = reg.predict(test_X)
        # y_pred = reg2.predict(test_df[cols])
        # result.append(y_pred)
        #
        #
        # ##################
        # reg3.fit(X_train, y_train)
        #
        # y_pre = reg3.predict(X_test)
        # # print(np.sqrt(metrics.mean_squared_error(y_test, y_pre)))
        # # y_test=ss_y.inverse_transform(y_test)
        # # y_pre=ss_y.inverse_transform(y_pre)
        # compare(y_test, y_pre)
        # # y_pred = reg.predict(test_X)
        # y_pred = reg3.predict(test_df[cols])
        # # result.append(y_pred)
        #
        #
        # ##################
        # reg4.fit(X_train, y_train)
        #
        # y_pre = reg4.predict(X_test)
        # # print(np.sqrt(metrics.mean_squared_error(y_test, y_pre)))
        # # y_test=ss_y.inverse_transform(y_test)
        # # y_pre=ss_y.inverse_transform(y_pre)
        # compare(y_test, y_pre)
        # # y_pred = reg.predict(test_X)
        # y_pred = reg4.predict(test_df[cols])
        # # result.append(y_pred)

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

    testpredict["prediction_pay_price"] = testpredict.apply(lambda x: function(x.pay_price, x.prediction_pay_price), axis = 1)
    testpredict = testpredict[["user_id", "prediction_pay_price"]]
    testpredict = testpredict.sort_values('prediction_pay_price', ascending=False)
    testpredict.to_csv(save_path,index=0)


def function(a, b):
    if a>b:
        return a*12
    else:
        return b

if __name__ == "__main__":
    predict = model(train_path, test_path)
    generate_summit(predict, test_path, save_path)
