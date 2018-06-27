import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#from ggplot import diamonds
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
#from keras.models import Sequential,Model
#from keras.layers import Dense,Input
import seaborn as sns
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
#matplotlib.style.use('ggplot')
# Figures inline and set visualization style
#%matplotlib inline

# Data Analysis
sns.set()
data_df=pd.read_csv('train.csv',encoding='utf-8')
dt_df=pd.read_csv('test.csv',encoding='utf-8')




#feature Engineering

cat_area=pd.get_dummies(data_df['condition'])
cat_area_t=pd.get_dummies(dt_df['condition'])
data_df=data_df.join(cat_area)
dt_df=dt_df.join(cat_area_t)

print(data_df.dtypes)
data=data_df.drop(['item_id'],axis=1)
data=data.drop(['category_class'],axis=1)
predictor_lst2=list(data.select_dtypes(include=['uint8','int64'],exclude=['object','float64']))
print(predictor_lst2)
#predictor_lst=['price','sold_price','size','Fair','Good','Like New']

y=data_df['category_class']
X_train, X_test, y_train, y_test = train_test_split(data_df[predictor_lst2], y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



#model training


#without transfor data

lr=LogisticRegression()
#lr.fit(data_df[predictor_lst],data_df['category_class'])
lr.fit(X_train,y_train)
print("*** Accuracy no transformmation (Score) ***")
#print(lr.score(data_df[predictor_lst], data_df["category_class"]))
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))
y_pred=lr.predict(X_test)
#print(f1_score(y_train,lr.predict(X_train),average='macro'))
print(f1_score(y_test,lr.predict(X_test),average='macro'))

###standardization

from sklearn.preprocessing import StandardScaler
#X_train=data_df[predictor_lst].values
#y=data_df['category_class']
scaler = StandardScaler().fit(X_train)
X_rescaled=scaler.transform(X_train)
scale_test=StandardScaler().fit(X_test)
stdXtest=scale_test.transform(X_test)

lr=LogisticRegression()
lr.fit(X_rescaled,y_train)
print("*** Accuracy standardized(Score) ***")
print(lr.score(X_rescaled,y_train))
print("***Test Accuracy standardized(Score) ***")
print(lr.score(stdXtest,y_test))
y_pred=lr.predict(stdXtest)
print("**F1 score standardized(Score) ***")

print(f1_score(y_test,y_pred,average='macro'))



####normalization

from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalizedX = scaler.transform(X_train)
scaler_test=Normalizer().fit(X_test)
normXtest=scaler_test.transform(X_test)
lr=LogisticRegression()
lr.fit(normalizedX,y_train)

print("*** Accuracy normalized (Score) ***")
#print(lr.score(data_df[predictor_lst], data_df["category_class"]))
print(lr.score(normalizedX,y_train))
print(lr.score(normXtest,y_test))
y_pred=lr.predict(normXtest)
print(f1_score(y_test,y_pred,average='macro'))


