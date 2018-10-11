import pandas as pd

comman_path = r"../"

test = comman_path + r"data/original/tap_fun_test.csv"
train_one_path = comman_path + r"data/submit/result_pay_7_pay_45.csv"
train_two_path = comman_path + r"data/submit/result_pay_7_no_45.csv"
train_three_path = comman_path + r"data/submit/result_no_7_pay_45.csv"
train_four_path = comman_path + r"data/submit/result_no_7_no_45.csv"
save_path = comman_path + r"data/submit/result.csv"
one = pd.read_csv(train_one_path)
two = pd.read_csv(train_two_path)
three = pd.read_csv(train_three_path)
four = pd.read_csv(train_four_path)

result = one.append(two)
result = result.append(three)
result = result.append(four)

test_df = pd.read_csv(test)

test_df = pd.merge(test_df,result,on="user_id",how="left")
test_df[["user_id","prediction_pay_price"]].to_csv(save_path,index=0)