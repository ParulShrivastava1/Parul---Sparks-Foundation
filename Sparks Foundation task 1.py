#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Sparks Foundation
# Data Science and Business Analytics Intern
# 
# Author : Parul
# 
# Task 1 : Prediction using Machine learning
# 
# In this task we have to predict the percentage score of a student based on the number of hours studied . The task has two variables where the feature is the number of hours studied and the target value is the percentage score . this can be solved using simple linear regression .

# In[12]:


# Importing required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading data from remote link 
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# In[3]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show() 


# From the graph above , we can see that there is a positive linear relation between the number of hours studied and percentage of score.

# # Preapring the data
# The next step is to divide the data into "attributes"(inputs) and "labels"(outputs)

# In[4]:


### Indepent and Dependent features 
X = s_data.iloc[:, :-1].values
y = s_data.iloc[:, -1].values


# Now that we have our attributes and labels , the next step is to split this data into training and test sets.We will do this by using Scikit-Learn's built-in train_test_split()method:

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=0.2, random_state=0)


# # Training the Algorithm
# We have split our data into trainig and testing sets, and now is finally the time to train our algorithm.

# In[6]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training complete.")


# In[7]:


# Plotting the regression line 
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X,y)
plt.plot(X, line);
plt.show() 


# # Making predictions
# Now that we have trained our algorithm , it's time to make some predictions.

# In[8]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[9]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
df


# In[10]:


# We can also test with our own data 
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[11]:


from sklearn import metrics
print('Mean Absolute Error:',
     metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




