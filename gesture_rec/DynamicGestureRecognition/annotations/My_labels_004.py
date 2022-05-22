import pandas as pd

# read train csv into a pandas dataframe  读取train CSV到pandas数据帧
training_labels_df = pd.read_csv('jester-v1-train.csv' , header=None,sep = ";", index_col=False)
# read validation csv into a pandas dataframe
validation_labels_df = pd.read_csv('jester-v1-validation.csv' , header=None,sep = ";", index_col=False)
# read the labels csv into a pandas dataframe
labels_df = pd.read_csv('jester-v1-labels.csv' , header=None,sep = ";", index_col=False)

print("Train size: " + str(training_labels_df.size))
print("Validation size: " + str(validation_labels_df.size))
print("Labels size: " + str(labels_df.size))

# names of labels to include in our filtered labels  选择的标签
targets_name = [
    "Swiping Left",
    "Swiping Right",
    "Pushing Hand Away",
    "Pulling Hand In",
    "No gesture"
    ]

# 只保留targets_name列表中出现的标签
# training_labels_df[0] 为 train.csv中的第一列数值
# training_labels_df[1] 为 train.csv中的第二列标签
# training_labels_df[1].isin(targets_name) Series中的标签是否在targets_name列表中，返回值类型为 bool,如果为 true 则保留
training_labels_filtered = training_labels_df[training_labels_df[1].isin(targets_name)]
validation_labels_filtered = validation_labels_df[validation_labels_df[1].isin(targets_name)]
labels_filtered = labels_df[labels_df[0].isin(targets_name)]

print("Train Filtered size: " + str(training_labels_filtered.size))
print("Validation Filtered size: " + str(validation_labels_filtered.size))
print("Labels Filtered size: " + str(labels_filtered.size))

# 保存为 csv 格式文件
training_labels_filtered.to_csv('jester-v1-train-My004.csv',header=False,sep=";",index=False)
validation_labels_filtered.to_csv('jester-v1-validation-My004.csv',header=False,sep=";",index=False)
labels_filtered.to_csv('jester-v1-labels-My004.csv',header=False,sep=";",index=False)