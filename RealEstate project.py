#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('data4project.csv')


# In[3]:


df = df.drop(columns = 'sqft')


# In[4]:


df.dtypes


# In[5]:


df.describe()


# In[6]:


correlations = df.corr()
correlations


# In[7]:


fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations, vmin=-1, vmax = 1)
fig.colorbar(cax)
ticks=np.arange(0, 11,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_yticklabels(correlations.columns)
ax.set_title('Overall Correlation Matrix')


# In[8]:


two_bedrooms = df[df['bedrooms'] < 3]
two_bedrooms
two_bedrooms


# In[9]:


correlations_two_bedrooms = two_bedrooms.corr()
correlations_two_bedrooms


# In[10]:


fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations, vmin=-1, vmax = 1)
fig.colorbar(cax)
ticks=np.arange(0, 11,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_yticklabels(correlations.columns)
ax.set_title('2-Bedroom Correlation Matrix')


# In[11]:


two_bedrooms.plot(kind='density', subplots=True, layout=(4,4), sharex=False, figsize=(20,20))


# In[28]:


sns.scatterplot(x=two_bedrooms['days_on_market'], y=two_bedrooms['price_sold'], hue = two_bedrooms['school_dist'], size = two_bedrooms['school_dist'])


# In[13]:


two_bedrooms.describe()


# In[14]:


plt.hist(two_bedrooms['price_sold'])


# In[15]:


sns.distplot(two_bedrooms['days_on_market'])


# In[16]:


per_zipcode = two_bedrooms.groupby('zipcode').mean().reset_index()
per_zipcode = per_zipcode.sort_values('price_sold', ascending = True)
per_zipcode


# In[17]:


fig, ax = plt.subplots()
sns.barplot(x = 'zipcode', y = 'price_sold', data = per_zipcode, capsize = 0.2, ci = 'sk')
ax.set_xticklabels(per_zipcode['zipcode'], rotation=90)
ax.set_title('Average Price per Zipcode')


# In[18]:


def adjs_inflation (row):
    if row['year'] <= 2005.0:
        return row['price_sold']/1.01
    if row['year'] <= 2006:
        return row['price_sold']/1.02/1.01
    return 1.3


# In[19]:


df['real_price_sold'] = df.apply (lambda row: adjs_inflation(row), axis=1)


# In[20]:


df


# Price per year per rooms and then adjust it to inflation

# In[21]:


per_yr_room = df.groupby(['year', 'bedrooms']).mean()
per_yr_room


# In[22]:


per_room = per_yr_room['price_sold']
per_room = per_room.unstack()
per_room.head()


# In[23]:


fig, ax1 = plt.subplots()
plt.plot(per_room)


# In[24]:


my_plot = per_room.plot(kind='line', stacked=False, title="Price Per Year")
my_plot.set_xlabel("Customers")
my_plot.set_ylabel("Sales ($)")
my_plot.legend(loc='best', ncol=4)


# In[32]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


# In[40]:


X = df[['bedrooms', 'days_on_market', 'year', 'yearbuilt', 'baths', 'school_dist']]
y = df[['price_sold']]


# In[49]:


X.isna().sum()
X1 = X.dropna()
X1.isna().sum()


# In[57]:


testdropna = df.dropna()
X2 = testdropna[['bedrooms', 'days_on_market', 'year', 'yearbuilt', 'baths', 'school_dist']]
y2 = testdropna[['price_sold']]


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)


# In[68]:


lasso = Lasso(alpha=0.1, normalize = True)
names = X2.columns
lasso_coef = lasso.fit(X2, y2).coef_
plt.plot(range(len(names)), lasso_coef)
plt.xticks(range(len(names)), names, rotation = 60)
plt.ylabel('Coefficients')


# In[69]:


lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)


# In[70]:


#Issues with this model should include lack of property size and a one-fits all approach to creating the regression
#Next steps should include narrowing down the properties into smaller data sets filtering by bedroom and/or bath ranges

