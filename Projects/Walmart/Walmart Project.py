#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")


# In[3]:


walmart_data = pd.read_csv('Walmart.csv')
walmart_data.sample(10)


# # Examine the Data Structure

# In[70]:


warnings.filterwarnings("ignore", category=FutureWarning)
walmart_data.describe(include='all')


# In[4]:


walmart_data.info()


# In[71]:


warnings.filterwarnings("ignore", category=UserWarning)
walmart_data['Date'] = walmart_data['Date'].astype('datetime64')


# In[6]:


walmart_data.info()


# In[7]:


walmart_data.describe()


# In[8]:


## Checking size of Dataset


# In[9]:


walmart_data.shape


# ## Checking for null values

# In[10]:


walmart_data.isnull().sum()


# # Clean and Preprocess the Data

# ### Checking for outliers

# In[73]:


fig,axis = plt.subplots(4,figsize=(16,16))
x=walmart_data[['Temperature','Fuel_Price','CPI','Unemployment']]
for i, column in enumerate(x):
    sns.boxplot(walmart_data[column],ax=axis[i])
warnings.filterwarnings("ignore", category=FutureWarning)


# In[75]:


walmart_data[(walmart_data['Unemployment']<4.5) | (walmart_data['Unemployment']>11)].shape

# Since data is larger so we cannot drop this. Treating the outliers here.
# In[76]:


# Using IQR Method
def thr_min_max(col):
    p25= walmart_data[col].quantile(0.25)
    p75=walmart_data[col].quantile(0.75)
    IQR=p75-p25
    
    thr_min,thr_max = p25 - 1.5* IQR , p75+1.5*IQR
    return thr_min,thr_max


def treating(val):
    if(val<thr_min):
        return thr_min
    elif(val>thr_max):
        return thr_max
    else:
        return val


# In[77]:


thr_min,thr_max = thr_min_max('Unemployment')
print(thr_min,thr_max)
walmart_data['Unemployment_treated'] = walmart_data['Unemployment'].apply(treating)


# In[78]:


walmart_data.sample(7)


# In[15]:


walmart_data[(walmart_data['Temperature']<5)]


# In[16]:


# Only one column with outlier of Temperature. So dropping it off.

walmart_data = walmart_data.drop(walmart_data[(walmart_data['Temperature']<5)].index)


# In[17]:


walmart_data.sample(10)


# # Data Visualisation

# In[18]:


walmart_data['Store'].unique()
# 45 stores data is present


# In[79]:


# Which store has maximum sales 

px.bar(data_frame=walmart_data,x='Store',y='Weekly_Sales')


# In[20]:


# Here we can see store 20 has max sales and then Store 4 has 2nd highest sale


# In[21]:


# store having maximum standard deviation
store_std = walmart_data.groupby('Store')['Weekly_Sales'].std().reset_index()
px.bar(x=store_std['Store'],y=store_std['Weekly_Sales'])


# In[22]:


# Thus, the store which has maximum standard deviation is store number 14, which means that the sales of Store 14 varies the most


# In[80]:


# checking which month has high sales
warnings.filterwarnings("ignore", category=DeprecationWarning)
px.bar(x=walmart_data['Date'],y=walmart_data['Weekly_Sales'])


# In[24]:


# We can observer from above graph that sales for Month of December is majorly the maximum sales in a year.


# In[25]:


# Checking the sales for year 2010

data = walmart_data[(walmart_data['Date']> '2010-01-01') & (walmart_data['Date']< '2011-01-01')].groupby('Store')['Weekly_Sales'].sum()
data=pd.DataFrame(data).reset_index()
data


# In[26]:


px.bar(x=data['Store'], y=data['Weekly_Sales'])

# For yr 2010 store 14 had max sales
# In[27]:


# Checking the effect of holiday on Sales
data = walmart_data.groupby('Holiday_Flag')['Weekly_Sales'].mean()
data=pd.DataFrame(data).reset_index()
data


# In[82]:


warnings.filterwarnings("ignore", category=DeprecationWarning)
px.bar(x=data['Holiday_Flag'],y=data['Weekly_Sales'],labels={'x':'Holiday or not','y':'Total Sales'})


# In[29]:


# We can see that Total Sales are higher during Holidays
walmart_data.head()


# In[30]:


#Year-wise Monthly Sales

walmart_data['Year'] = pd.to_datetime(walmart_data['Date']).dt.year
walmart_data['Month'] = pd.to_datetime(walmart_data['Date']).dt.month_name()
walmart_data['Day'] = pd.to_datetime(walmart_data['Date']).dt.day
year_month_sales = walmart_data.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
print(year_month_sales.sort_values(by=['Year', 'Month']))


# In[31]:


px.bar(data_frame=year_month_sales, x='Month',y='Weekly_Sales',color='Year')


# In[32]:


px.bar(data_frame=year_month_sales, x='Year',y='Weekly_Sales',color='Year')


# In[83]:


# Check effect of Fuel_Price on Weekly_Sales
warnings.filterwarnings("ignore", category=DeprecationWarning)

px.scatter(x=walmart_data['Fuel_Price'],y=walmart_data['Weekly_Sales'])


# In[34]:


# From this we can say there is no direct relation bw these two.


# In[35]:


# relation bw temperature and sales

px.scatter(x=walmart_data['Temperature'],y=walmart_data['Weekly_Sales'])


# In[36]:


px.bar(x=walmart_data['Temperature'],y=walmart_data['Weekly_Sales'],color='Weekly_Sales',data_frame=walmart_data)

# acc to above 2 graphs we can conclude that sales is comparatively higher when temperature is bw 25 to 55. Else the sales is comparatively lower
# In[37]:


# Heatmap
plt.figure(figsize=(8,8))
sns.heatmap(walmart_data.corr(),annot=True,fmt='.1f',linewidths=.8)


# In[85]:


# Standardization

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X = walmart_data[['Store','Fuel_Price','CPI','Unemployment','Year']]
Y = walmart_data['Weekly_Sales']
X_std = sc.fit_transform(X)
X_std


# In[91]:


# Linear Regression :

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, Y_train, Y_test = train_test_split(X_std,Y,test_size=0.2)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[92]:


reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_test_pred = reg.predict(X_test)
from sklearn import metrics
print('Accuracy of test data:',reg.score(X_test, Y_test)*100)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred)))
sns.scatterplot(Y_test_pred, Y_test)


# In[96]:


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
randomfrst = RandomForestRegressor()        
randomfrst.fit(X_train,Y_train)
Y_pred = randomfrst.predict(X_test)
print('Accuracy of test data:',randomfrst.score(X_test, Y_test)*100)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
sns.scatterplot(Y_pred, Y_test)


# In[97]:


# Decision Tree
from sklearn.tree import DecisionTreeRegressor
X_train, X_test, y_train, y_test = train_test_split(X_std,Y,test_size=0.2)
model = DecisionTreeRegressor()
model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print("Accuracy of Train data:", model.score(X_train,y_train)*100)
print('Accuracy of test data:',model.score(X_test, y_test)*100)
print("Mean Absolute Error: ",metrics.mean_absolute_error(y_test,y_pred_test))
print("Mean Square Error: ",metrics.mean_squared_error(y_pred_test,y_test))
print("Root Mean Square Error: ",np.sqrt(metrics.mean_squared_error(y_pred_test,y_test)))
sns.scatterplot(y_test,y_pred_test)

# Here, Linear Regression is not an appropriate model to use which is clear from it's low accuracy. 
# However, Random Forest Regression gives higher accuracy , so, it is the best model to forecast demand.
We can conclude that RandomForest model is best suited for our predictions.
# In[51]:


walmart_data['Weekly_Sales'].plot()
# This is seasonal data


# In[54]:


from statsmodels.tsa.stattools import adfuller
sales_data = walmart_data['Weekly_Sales']
result = adfuller(sales_data)
p_value = result[1]
print("P_value: ",p_value)
if p_value < 0.05:
    print("The data is stationary.")
else:
    print("The data is not stationary.")    


# In[ ]:


# Our data is stationary and seasonal. We can use SARIMA.


# In[62]:


import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

import statsmodels.api as sm

unique_stores = walmart_data['Store'].unique()

order = (1, 1, 1) 
seasonal_order = (1, 1, 1, 12)

for store_id in unique_stores:
    store_data = walmart_data[walmart_data['Store'] == store_id]
    model = sm.tsa.statespace.SARIMAX(store_data['Weekly_Sales'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=12)
    print(f"Store ID: {store_id}")
    print(forecast)
    print("---------------------")


# In[ ]:




