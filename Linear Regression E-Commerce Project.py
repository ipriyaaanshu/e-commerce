
# coding: utf-8

# # Linear Regression : E - Commerce Project
# 
# An Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website.
# 
# (it's fake, don't worry I didn't give you real credit card numbers or emails).

# ## Imports
# ** Import pandas, numpy, matplotlib, and seaborn. Then set %matplotlib inline 
# (I'll import sklearn as I need it.)**

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Getting the Data
# 
# I'll work with the Ecommerce Customers csv file from the company. It has Customer info, such as Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# 
# ** I'll Read in the Ecommerce Customers csv file as a DataFrame called df (this is what I usually use).**

# In[9]:


df = pd.read_csv('Ecommerce Customers')


# **Check the head of customers, and check out its info() and describe() methods.**

# In[10]:


df.head()


# In[11]:


df.describe()


# In[12]:


df.info()


# ## Exploratory Data Analysis
# 
# **Let's explore the data!**
# 
# For the rest of the project I'll only be using the numerical data of the csv file.
# ___
# ** I'll Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[13]:


sns.set_palette('GnBu_d')
sns.set_style('whitegrid')


# In[33]:


sns.jointplot(data=df,x='Time on Website',y='Yearly Amount Spent')


# ** I'll Do the same but with the Time on App column instead. **

# In[15]:


sns.jointplot(data=df,x='Time on App',y='Yearly Amount Spent')


# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# In[16]:


sns.jointplot(data=df,x='Time on App',y='Length of Membership',kind='hex')


# **Let's explore these types of relationships across the entire data set by using pairplot.**
# 
# The pairplot will give me all the correlations at once.

# In[17]:


sns.pairplot(df)


# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**
# ### Length of Membership

# ** Now I'll create a linear model plot of Yearly Amount Spent vs. Length of Membership. **

# In[18]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df)


# ## Training and Testing Data
# 
# Now that I've explored the data a bit, let's go ahead and split the data into training and testing sets.
# ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **

# In[19]:


X = df[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]


# In[20]:


y = df['Yearly Amount Spent']


# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the Model
# 
# Now its time to train this model on our training data!
# 
# ** Import LinearRegression from sklearn.linear_model **

# In[23]:


from sklearn.linear_model import LinearRegression


# **Create an instance of a LinearRegression() model named lm.**

# In[24]:


lm = LinearRegression()


# ** Train/fit lm on the training data.**

# In[25]:


lm.fit(X_train,y_train)


# **Print out the coefficients of the model**

# In[26]:


lm.coef_


# ## Predicting Test Data
# Now that I have fit this model, let's evaluate its performance by predicting off the test values!
# 
# ** Use lm.predict() to predict off the X_test set of the data.**

# In[27]:


predictions = lm.predict(X_test)


# ** Create a scatterplot of the real test values versus the predicted values. **

# In[28]:


plt.scatter(x=y_test,y=predictions)


# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.**

# In[29]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## Residuals
# 
# We have got a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
# 
# **I will plot a histogram of the residuals and make sure it looks normally distributed.**

# In[30]:


sns.distplot((y_test - predictions))


# ## Conclusion
# We still want to figure out the answer to the original question, do they need to focus their efforts on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.

# In[31]:


cdf = pd.DataFrame(lm.coef_,X_train.columns,columns=['Coeffecient'])
cdf


# In[32]:


metrics.r2_score(y_test,predictions)


# ** How can you interpret these coefficients? **

# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in **Avg. Session Length** is associated with an **increase of 25.98 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Time on App** is associated with an **increase of 38.59 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Time on Website** is associated with an **increase of 0.19 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Length of Membership** is associated with an **increase of 61.27 total dollars spent**.

# **Do you think the company should focus more on their mobile app or on their website?**

# 
# This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better. This sort of answer really depends on the other factors going on at the company, I would probably want to explore the relationship between Length of Membership and the App or the Website before coming to a conclusion!
# 
