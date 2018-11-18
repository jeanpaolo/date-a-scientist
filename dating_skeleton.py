import datetime
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# ##################
# ### Regression ###
# ##################
print("Starting Regression Run")
os.chdir("C:\Users\paolo.saul\Downloads\codecademy\capstone_starter_date_a_scientist")
df = pd.read_csv("profiles.csv")

#
# regression - calculate income based on essay length, height, age, or ethnicity
#

# exploring the dataset
df['signsimple'] = df['sign'].str.split(' ').str[0]
df.signsimple.value_counts().plot.bar()
#plt.show()

df['offspringsimple'] = df['offspring'].str.split(',').str[0]
df.offspringsimple.value_counts().plot.bar()
#plt.show()


# augmenting the dataset
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
all_essays = df[essay_cols].replace(np.nan, ' ', regex = True)
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis = 1)
df["essay_len"] = all_essays.apply(lambda x: len(x))
df["all_essays"] = all_essays
# removing unusable rows
df = df[ df["essay_len"] != np.nan]
df = df[ df["essay_len"] > 0]
df = df [ df["income"] > -1]
df = df[df.income != np.nan]
df["height"] = df["height"].replace(np.nan, 0, regex=True)
df["height"] = df["height"].fillna(0)
# clean up ethnicity to only get the first associated ethnicity
df["simple_ethnic"] = df["ethnicity"].replace(np.nan, "other",regex=True)
df["simple_ethnic"] = df["simple_ethnic"].str.split(", ").str[0]
df["simple_ethnic"] = df["simple_ethnic"].replace(np.nan, "other")
df["ethcode"] = df["simple_ethnic"].astype("category").cat.codes
df["ethcode"] = df["ethcode"].replace(np.nan, 0)
df = df[df.age != np.nan]
df["age"] = df["age"].dropna()


# for regression we scale the numeric values of our selected features
feature_data = df[['age','height','essay_len','ethcode']]
x = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)


X_train, X_test, y_train, y_test = train_test_split(feature_data, df.income)

print("Starting Regression Training")
timestart = datetime.datetime.now()

# regressors
classifier = LinearRegression()
classifier.fit(X_train, y_train)

print("Regression Results:")
print("Regression Score: {}".format(classifier.score(X_test, y_test)))
timestop = datetime.datetime.now()
print("Regression Training Time: {}".format(timestop - timestart))



# ####################
# ## Classification ##
# ####################
print("\n\nStarting Classification Run")
os.chdir("C:\Users\paolo.saul\Downloads\codecademy\capstone_starter_date_a_scientist")
df = pd.read_csv("profiles.csv")

#
# classification - predict sex from essay contents
#

# exploring the dataset
#df['signsimple'] = df['sign'].str.split(' ').str[0]
#df.signsimple.value_counts().plot.bar()
#plt.show()

#df['offspringsimple'] = df['offspring'].str.split(',').str[0]
#df.offspringsimple.value_counts().plot.bar()
#plt.show()


# augmenting the dataset
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
all_essays = df[essay_cols].replace(np.nan, ' ', regex = True)
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis = 1)
df["essay_len"] = all_essays.apply(lambda x: len(x))
df["all_essays"] = all_essays
# removing unusable rows
df = df[ df["essay_len"] != np.nan]
df = df[ df["essay_len"] > 0]
df = df[ df["sex"] != np.nan]
df = df [ df["income"] > -1]

# for classifiers with essay contents, we need to count the words of all the essays
counter = CountVectorizer()

print("Starting Counter Fit")
counter.fit(df.all_essays)

X_train, X_test, y_train, y_test = train_test_split(df.all_essays, df.sex)
train_counts = counter.transform(X_train)
test_counts = counter.transform(X_test)

print("Starting Classification Training")
timestart = datetime.datetime.now()

# classifiers
classifier = MultinomialNB()
#classifier = SVC(kernel = 'rbf', gamma = 0.1)
classifier.fit(train_counts, y_train)
predict_test = classifier.predict(test_counts)

print("Classification Results:")
print("Classification Score: {}".format(classifier.score(test_counts, y_test)))
print(accuracy_score(predict_test, y_test))
print(recall_score(predict_test, y_test, average='micro'))
print(precision_score(predict_test, y_test, average='micro'))
print(f1_score(predict_test, y_test, average='micro'))
timestop = datetime.datetime.now()
print("Classification Training Time: {}".format(timestop - timestart))
