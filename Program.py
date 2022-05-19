### Multinomial Regression ####
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

education = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/mdata.csv")

#Removing of unnecessary columns
education1 = education.drop(["serial no","id"], axis = 1)
education1

education1.columns = "gender","ses","schtyp","prog","read","write","math","science","honors"
education1.head() # Shows first 5 columns of the dataset

education1.describe()

# Count of variables
education1.prog.value_counts()
# academic=105 , vocation = 50 and general = 45
education1.gender.value_counts()
education1.ses.value_counts()
education1.schtyp.value_counts()
education1.honors.value_counts()

# Rearrange the order of the variables
education = education1.iloc[:, [3, 0,1,2,4,5,6,7,8]]
education.columns

# Creation of Dummy variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
education['gender'] = le.fit_transform(education['gender'])
education['ses'] = le.fit_transform(education['ses'])
education['schtyp'] = le.fit_transform(education['schtyp'])
education['honors'] = le.fit_transform(education['honors'])

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "prog", y = "read", data = education)
sns.boxplot(x = "prog", y = "write", data = education)
sns.boxplot(x = "prog", y = "math", data = education)
sns.boxplot(x = "prog", y = "science", data = education)

# Scatter plot for each categorical choice of car
sns.stripplot(x = "prog", y = "read", jitter = True, data = education)
sns.stripplot(x = "prog", y = "write", jitter = True, data = education)
sns.stripplot(x = "prog", y = "math", jitter = True, data = education)
sns.stripplot(x = "prog", y = "science", jitter = True, data = education)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(education) # Normal
sns.pairplot(education, hue = "prog") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
education.corr()

train, test = train_test_split(education, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:],train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions
# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 

# Conclusion
# as the test accuracy = 0.625 and train accuracy = 0.6625 , which indiactes that there is no major variance 
#which indicate the model is Right Fit