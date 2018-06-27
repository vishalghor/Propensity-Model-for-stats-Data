
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
import matplotlib.pyplot as plt
import csv

# Data Analysis
sns.set()
data_df=pd.read_csv('train.csv',encoding='utf-8')
dt_df=pd.read_csv('test.csv',encoding='utf-8')

print(data_df.describe())
print(data_df.head())

corr=data_df.corr()
print(corr)

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
plt.show(sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns))

plt.show(data_df['category_class'].hist(bins=50))

diag=pd.crosstab(data_df['condition'],data_df['category_class'])
plt.show(diag.plot(kind='bar', stacked=True, color=['red','blue','green','yellow','orange'], grid=False))

#feature Engineering

cat_area=pd.get_dummies(data_df['condition'])
data_df=data_df.join(cat_area)


print(data_df.dtypes)
data=data_df.drop(['item_id'],axis=1)
data=data.drop(['category_class'],axis=1)
predictor_lst2=list(data.select_dtypes(include=['uint8','int64'],exclude=['object','float64']))

print(predictor_lst2)

y=data_df['category_class']
X_train, X_test, y_train, y_test = train_test_split(data_df[predictor_lst2], y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


###standardization

from sklearn.preprocessing import StandardScaler
#X_train=data_df[predictor_lst].values
#y=data_df['category_class']
scaler = StandardScaler().fit(X_train)
X_rescaled=scaler.transform(X_train)
scale_test=StandardScaler().fit(X_test)
stdXtest=scale_test.transform(X_test)

dummy_y = np_utils.to_categorical(y_train)
#print(np.unique(dummy_y))
model.add(Dense(16,input_dim=16,activation='relu'))
model.add(Dense(12,input_dim=16,activation='relu'))
model.add(Dense(10,input_dim=12,activation='relu'))
model.add(Dense(5,input_dim=10,activation='relu'))
model.add(Dense(5,input_dim=5,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=0)

model.fit(X_rescaled,dummy_y,epochs=50,batch_size=10)
#np_utils.to_categorical(y_train)
score=model.evaluate(X_rescaled,dummy_y)
print(score)

#*********************accuracy of 61% which is less than GradientBoostingClassifier*********************************