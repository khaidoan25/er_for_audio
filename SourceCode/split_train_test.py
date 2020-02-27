import pandas as pd

df = pd.read_csv("train_label.csv")
df_sort = df.sort_values(by='Label')

df_neutral = df_sort.iloc[:764]
df_happy = df_sort.iloc[764:1658]
df_sad = df_sort.iloc[1658:2551]
df_anger = df_sort.iloc[2551:3444]
df_fear = df_sort.iloc[3444:4337]
df_disgust = df_sort.iloc[4337:5230]

test_1 = df_neutral.iloc[:170].append(df_happy.iloc[:175]).append(df_sad.iloc[:175]).append(df_anger.iloc[:175]).append(df_fear.iloc[:175]).append(df_disgust.iloc[:175]).sample(frac=1)
train_1 = df[~df["File"].isin(test_1["File"])]
test_2 = df_neutral.iloc[170:341].append(df_happy.iloc[175:351]).append(df_sad.iloc[175:351]).append(df_anger.iloc[175:351]).append(df_fear.iloc[175:351]).append(df_disgust.iloc[175:351]).sample(frac=1)
train_2 = df[~df["File"].isin(test_2["File"])]
test_3 = df_neutral.iloc[341:512].append(df_happy.iloc[351:522]).append(df_sad.iloc[351:522]).append(df_anger.iloc[351:522]).append(df_fear.iloc[351:522]).append(df_disgust.iloc[351:522]).sample(frac=1)
train_3 = df[~df["File"].isin(test_3["File"])]

train_1.columns = ["File", "Label"]
train_1.to_csv("train_1.csv", index=False)
test_1.columns = ["File", "Label"]
test_1.to_csv("test_1.csv", index=False)
train_2.columns = ["File", "Label"]
train_2.to_csv("train_2.csv", index=False)
test_2.columns = ["File", "Label"]
test_2.to_csv("test_2.csv", index=False)
train_3.columns = ["File", "Label"]
train_3.to_csv("train_3.csv", index=False)
test_3.columns = ["File", "Label"]
test_3.to_csv("test_3.csv", index=False)