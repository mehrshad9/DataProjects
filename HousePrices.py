#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
df_test2=df_test.copy()


# In[ ]:


df_train.set_index('Id' , inplace=True)
df_test.set_index('Id' , inplace=True)


# In[ ]:


df_train.info()


# In[ ]:


y=df_train.pop('SalePrice')


# In[ ]:


nancol=df_train.isna().sum().sort_values(ascending = False) [1:20]


# In[ ]:


df_train.isnull().sum().sort_values(ascending = False) [1:20]


# In[ ]:


nancol.index


# In[ ]:


dropcol=[]
for i in nancol.index:
    if nancol.loc[i]/len(df_train) >0.4:
        dropcol.append(i)


# In[ ]:


dropcol


# In[ ]:


dropcol.append('PoolQC')


# In[ ]:


dropcol


# In[ ]:


df_train.drop(dropcol , axis= 1 , inplace= True)
df_test.drop(dropcol , axis= 1 , inplace= True)


# In[ ]:


df_train


# In[ ]:


num_col=df_train._get_numeric_data().columns


# In[ ]:


cat_col=list(set(df_train) -set(num_col))


# In[ ]:


from sklearn.impute import SimpleImputer
si_float=SimpleImputer(missing_values=np.nan , strategy='mean')
si_float2=SimpleImputer(missing_values=np.nan , strategy='mean')


# In[ ]:


df_train[num_col]=si_float.fit_transform(df_train[num_col])
df_test[num_col]=si_float2.fit_transform(df_test[num_col])


# In[ ]:


si_cat=SimpleImputer(missing_values=np.nan , strategy='most_frequent')
df_train[cat_col]=si_cat.fit_transform(df_train[cat_col])
si_cat2=SimpleImputer(missing_values=np.nan , strategy='most_frequent')
df_test[cat_col]=si_cat2.fit_transform(df_test[cat_col])


# In[ ]:


df_train.isnull().sum().sort_values(ascending=False)[1:20]


# In[ ]:


ordinal_columns = ['Street','LotShape','LandContour','Utilities','LandSlope','BldgType','HouseStyle','ExterQual','ExterCond','BsmtQual','BsmtCond',
                   'BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','KitchenQual','Functional','GarageFinish','GarageQual','GarageCond',
                   'PavedDrive']


# In[ ]:


df_train['train']=1
df_test['train']=0


# In[ ]:


df=pd.concat([df_train , df_test] , axis=0 , sort=False)
df.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le={}
for col in ordinal_columns:
    le[col]=LabelEncoder()
    df[col]=le[col].fit_transform(df[col])


# In[ ]:


df[ordinal_columns]


# In[ ]:


df2=pd.get_dummies(df)


# In[ ]:


df_train=df2[df2['train'] > 0.5]


# In[ ]:


df_test=df2[df2['train'] < 0.5]


# In[ ]:


#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#df_train[num_col]=sc.fit_transform(df_train[num_col])
#df_test[num_col]=sc.fit_transform(df_test[num_col])


# In[ ]:


from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte= train_test_split(df_train , y, test_size=0.2 , random_state=10)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
lr=LinearRegression()
lr.fit(xtr , ytr)
y_pred=lr.predict(xte)
print(mean_squared_error(y_pred,yte))
print(r2_score(y_pred,yte))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
lr=RandomForestRegressor(n_estimators=500)
lr.fit(xtr , ytr)
y_pred=lr.predict(xte)
print(mean_squared_error(y_pred,yte))
print(r2_score(y_pred,yte))
pred=lr.predict(df_test)
sub_df = pd.DataFrame({'Id': df_test2['Id'], 'SalePrice': pred})
sub_df.to_csv('submit31.csv', index=False)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score
lr=KNeighborsRegressor(n_neighbors=5)
lr.fit(xtr , ytr)
y_pred=lr.predict(xte)
print(mean_squared_error(y_pred,yte))
print(r2_score(y_pred,yte))
pred=lr.predict(df_test)
sub_df = pd.DataFrame({'Id': df_test2['Id'], 'SalePrice': pred})
sub_df.to_csv('submit32.csv', index=False)


# In[ ]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score
lr=XGBRegressor()
lr.fit(xtr , ytr)
y_pred=lr.predict(xte)
print(mean_squared_error(y_pred,yte))
print(r2_score(y_pred,yte))


# In[ ]:


pred=lr.predict(df_test)
sub_df = pd.DataFrame({'Id': df_test2['Id'], 'SalePrice': pred})
sub_df.to_csv('submit33.csv', index=False)


# In[ ]:


# now with feature selection:


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=20)
#xtr=lda.fit_transform(xtr,ytr)


# In[ ]:


xtr.shape


# In[ ]:





# In[ ]:




