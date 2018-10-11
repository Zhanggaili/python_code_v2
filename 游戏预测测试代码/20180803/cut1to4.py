# -*- coding: utf-8 -*-
# @Date     :2018/7/31 13:27
# @Author   :cq_yang
# @:Describe:

"""
第一类：前7天给了45天给了
第二类：前7天给了45天没给
第三类：前7天没给45天给了
第四类：前7天没给45天没给
"""
from __future__ import division
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
from pandas.core.frame import DataFrame
warnings.filterwarnings('ignore')


comman_path = r"../"

train_path = comman_path + r"data/original/tap_fun_train.csv"
test_path = comman_path + r"data/original/tap_fun_test.csv"

train_one_path = comman_path + r"data/original/train_pay_7_pay_45.csv"
train_two_path = comman_path + r"data/original/train_pay_7_no_45.csv"
train_three_path = comman_path + r"data/original/train_no_7_pay_45.csv"
train_four_path = comman_path + r"data/original/train_no_7_no_45.csv"

test_one_path = comman_path + r"data/original/test_pay_7_pay_45.csv"
test_two_path = comman_path + r"data/original/test_pay_7_no_45.csv"
test_three_path = comman_path + r"data/original/test_no_7_pay_45.csv"
test_four_path = comman_path + r"data/original/test_no_7_no_45.csv"


def split_train_to_4(df):
    # df = pd.read_csv(data_path)
    # 提出异类样本

    df["leibie"] = 0
    df["leibie"] = df.apply(lambda x: 1 if (x["pay_price"] < x["prediction_pay_price"] and  x["pay_price"] != 0) else x["leibie"] ,axis=1)
    print(1)
    df["leibie"] = df.apply(lambda x: 2 if x["pay_price"] == x["prediction_pay_price"] else x["leibie"],axis=1)
    print(2)
    df["leibie"] = df.apply(lambda x: 3 if (x["pay_price"] < x["prediction_pay_price"] and  x["pay_price"]==0) else x["leibie"],axis=1)
    print(3)
    df["leibie"] = df.apply(lambda x: 4 if x["prediction_pay_price"] == 0 else x["leibie"],axis=1)
    print(4)
    df.to_csv(comman_path + r"data/original/tap_fun_train_tiny_4.csv",index=0)
    return df


def model(train_path,test_path):
    train_df = pd.read_csv(train_path)
    train_df = split_train_to_4(train_df)
    test_df = pd.read_csv(test_path)

    train_y = train_df["leibie"].values

    cols = [x for x in test_df.columns if x not in ["user_id", 'register_time']]
    train_x = train_df[cols]
    print(train_x.head(10))
    print(train_x.columns)

    test_x = test_df[cols]
    print(test_x.columns)
    print(test_x.head(10))
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    train_X = ss_X.fit_transform(train_x)
    test_X = ss_X.transform(test_x)
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=2008)
    model = xgb.XGBClassifier(booster= 'gbtree',
                              objective='multi:softmax',  # 多分类的问题
                              num_class= 4,
                              gamma=0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
                              max_depth= 12,               # 构建树的深度，越大越容易过拟合
                              # "lambda" = 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                              subsample= 0.7,              # 随机采样训练样本
                              colsample_bytree= 0.7,       # 生成树时进行的列采样
                              min_child_weight= 3,
                              silent=1,                   # 设置成1则没有运行信息输出，最好是设置为0.
                              eta=0.001,                  # 如同学习率
                              seed= 1000,
                              nthread= 4, )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(len(pred))
    print(sum(pred==y_test))
    print(sum(pred==y_test)/len(pred))
    ans = model.predict(X_test)
    # 对测试集进行预测

    ans = model.predict(test_X)
    test_df["leibie"] = ans

    return train_df, test_df


def save(df, p1, p2, p3, p4):
    p1_df = df.loc[df["leibie"] == 1]
    p2_df = df.loc[df["leibie"] == 2]
    p3_df = df.loc[df["leibie"] == 3]
    p4_df = df.loc[df["leibie"] == 4]
    p1_df.to_csv(p1,index=0)
    p2_df.to_csv(p2, index=0)
    p3_df.to_csv(p3, index=0)
    p4_df.to_csv(p4, index=0)


if __name__=="__main__":
    # split_train_to_4(train_path)
    train_df, test_df = model(train_path, test_path)
    save(train_df, train_one_path, train_two_path, train_three_path, train_four_path)
    save(test_df, test_one_path, test_two_path, test_three_path, test_four_path)
