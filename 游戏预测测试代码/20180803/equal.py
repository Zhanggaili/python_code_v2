# -*- coding: utf-8 -*-
# @Date     :2018/7/23 11:05
# @Author   :cq_yang
# @:Describe:model v 1.0
import lightgbm as lgbm
from xgboost import XGBRegressor
import pandas as pd
import datetime
import warnings
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
warnings.simplefilter("ignore")
# 是否对 y_train 进行标准化
Y_Standard = True

comman_path = r"../"

# train_path = comman_path + r"data/original/tap_fun_train.csv"
test_path = comman_path + r"data/original/test_pay_7_no_45_lgb.csv"

save_path = comman_path + r"data/submit/result_pay_7_no_45_lgb.csv"
            # r"result-%s.csv" % (datetime.datetime.now().strftime('%Y-%m-%d'))


#输出预测后的数据
def generate_summit(predict):
    testPredict = test_df.copy()
    testPredict["prediction_pay_price"] = predict
    testPredict = testPredict[["user_id", "prediction_pay_price"]]
    testPredict["prediction_pay_price"] = testPredict["prediction_pay_price"].apply(lambda x: x if x > 0 else 0)
    return testPredict


if __name__ == "__main__":
    test_df = pd.read_csv(test_path)
    liumang = test_df.pay_price.apply(lambda x:x)
    testPredict = generate_summit(np.array(liumang))
    testPredict.to_csv(save_path, index=False)
    