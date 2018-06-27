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

cat_var=['area_name','condition']
for var in cat_var:
    cat_area=pd.get_dummies(data_df[var])
    cat_area_t=pd.get_dummies(dt_df[var])
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

###gradient boosting algorithm

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
gmb=GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,min_samples_split=150,min_samples_leaf=50,max_depth=4,max_features='sqrt',subsample=0.8)
gmb.fit(X_rescaled,y_train)

print("****gmb score standardzed")
print(gmb.score(X_rescaled,y_train))
y_pred=gmb.predict(stdXtest)
print("***Test Accuracy standardized(Score) ***")

#model evaluation

print(gmb.score(stdXtest,y_test))
#print(f1_score(y_train,lr.predict(X_train),average='macro'))
print("***F1 score standardized(Score) ***")

print(f1_score(y_test,gmb.predict(stdXtest),average='macro'))




#normallized data
gmb=GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,min_samples_split=150,min_samples_leaf=50,max_depth=4,max_features='sqrt',subsample=0.8)
gmb.fit(X_rescaled,y_train)

print("****gmb score  normalized")
print(gmb.score(normalizedX,y_train))
y_pred=gmb.predict(normXtest)


#print(f1_score(y_train,lr.predict(X_train),average='macro'))
print(f1_score(y_test,gmb.predict(normXtest),average='macro'))


#without transform data
gmb=GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,min_samples_split=150,min_samples_leaf=50,max_depth=4,max_features='sqrt',subsample=0.8)
gmb.fit(X_train,y_train)

print("****gmb score without transform")
print(gmb.score(X_train,y_train))
y_pred=gmb.predict(X_test)


#print(f1_score(y_train,lr.predict(X_train),average='macro'))
print(f1_score(y_test,gmb.predict(X_test),average='macro'))






#predction

X_pred=np.array(dt_df[predictor_lst2])
print(X_pred)
#dt_df['category_class']=lr.predict(X_pred)
dt_df['category_class']=gmb.predict(X_pred)

dt_df.to_csv('output.csv',columns=['item_id','category_class'])


