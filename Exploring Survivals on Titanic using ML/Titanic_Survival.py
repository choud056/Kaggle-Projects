
# coding: utf-8

# In[1]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:

# Reading Data to training and test data sets

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[4]:

# Combinbing training and test data set for feature engineering

data = pd.concat([train_data,test_data])


# In[5]:

data.head()


# In[6]:

# Absorbing information from Name:

# Title
data['Title']= data['Name'].str.split('.').str[0].str.split(',').str[1].str.strip()
data['Title'].loc[data['Title']=='Ms']='Miss'
data['Title'].loc[data['Title']=='Mlle']='Miss'
data['Title'].loc[data['Title']=='Mme']='Mrs'
Rare = ['Rev','Dr','Col','Major','the Countess','Sir','Lady','Don','Jonkheer','Capt','Dona']
data['Title']=data['Title'].replace(Rare,'Rare')

# Surname

data['Surname'] = data['Name'].str.split('.').str[0].str.split(',').str[0].str.strip()
data['freq'] = data.groupby('Surname')['Surname'].transform('count')
data['Surname'].loc[data['freq']<3] = 'Others'

data=data.drop(['freq','Name','PassengerId'],axis=1)


# In[7]:

data.head()


# In[8]:

# Ticket information seems not needed so we can drop that too

data=data.drop('Ticket',axis=1)


# # A. Missing Value Imputation

# In[9]:

data.isnull().sum()


# # 1A.
# ## First Cabin Seems to have a lot of missing values, 
# ## let's work on that first

# In[10]:

# let's fill all empty values by 'U'(unknown)
data['Cabin']=data['Cabin'].fillna('U')
# Let's keep only the deck name 
data['Cabin']=data['Cabin'].str[0]


# 
# # 2A.
# # Embarked has only two missing values, can fill that with most frequent string 

# In[11]:

sns.countplot(x='Embarked',hue='Survived',data=data)


# In[12]:

# since 'S' is highest occuring value
data['Embarked'] = data['Embarked'].fillna('S')


# # 3A.
# # Fare only has one missing value, can fill that with median

# In[13]:

data['Fare']=data['Fare'].fillna(data['Fare'].dropna().median())


# # 4A.
# # Age has a lot of missing values, Let's see its distribution

# In[14]:

plt.hist(data['Age'].dropna(),50)


# In[15]:

# Age has a gaussian distribution, so most of the values are between mean+std and mean-std


# In[16]:

mean = data['Age'].dropna().mean()
std = data['Age'].dropna().std()
data['Age'][np.isnan(data['Age'])] = mean+std*np.random.rand(np.isnan(data['Age']).sum())


# In[17]:

plt.figure(figsize=(15,4))
sns.countplot(x='Age',hue='Survived',data=data)


# In[18]:

# It  seems that age<=16 has more survival rate compared to everyone else
# So, we can divide age in two groups 
data['Age'] = data['Age'].map(lambda x : 1 if (x<=16) else 2 )


# In[19]:

data.isnull().sum()


# In[20]:

# Worked with missing values


# # B. Making new features

# In[21]:

# Can Combine Parch and SibSp column to get the total family members


# # 1B
# # Family

# In[22]:

data['Family']=data['SibSp']+data['Parch']
sns.factorplot(x='Family',y='Survived',data=data)


# In[23]:

# From the plot it is clear that being alone or having a large family is not in favor of survival
# So, let's divide the family into three parts


# In[24]:

data['Family']=data['Family'].map(lambda x : 'Single' if (x==0) else 'Small' if (x<4) else 'Large')


# # 2B.
# # Mother

# In[25]:

data['Mother']=0
data['Mother'].loc[(data['Age']==2) & (data['Sex']=='female') & (data['Title']=='Mrs') & (data['Parch']>=1)]=1


# # 2C.
# # Child

# In[26]:

data['Child']=0
data['Child'].loc[(data['Age']==1)]=1


# In[27]:

data.head()


# # C. Label Encoding

# In[28]:

# Let's convert all the string variable to float or int


# In[29]:

from sklearn import preprocessing


# In[30]:

le1 = preprocessing.LabelEncoder()
le1.fit(data['Embarked'])
data['Embarked']=le1.transform(data['Embarked']) 


# In[31]:

le2 = preprocessing.LabelEncoder()
le2.fit(data['Sex'])
data['Sex']=le2.transform(data['Sex']) 


# In[32]:

le4 = preprocessing.LabelEncoder()
le4.fit(data['Title'])
data['Title']=le4.transform(data['Title'])


# In[33]:

le4 = preprocessing.LabelEncoder()
le4.fit(data['Surname'])
data['Surname']=le4.transform(data['Surname'])


# In[34]:

le4 = preprocessing.LabelEncoder()
le4.fit(data['Family'])
data['Family']=le4.transform(data['Family'])


# In[35]:

le4 = preprocessing.LabelEncoder()
le4.fit(data['Cabin'])
data['Cabin']=le4.transform(data['Cabin'])


# In[36]:

data.head()


# # D.  Random Forest Classification

# In[37]:

X_train_df = data.iloc[0:891,:].drop('Survived',axis=1)
y_train_df = train_data['Survived'] 

X_test_df = data.iloc[891:,:].drop('Survived',axis=1)


# In[38]:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0, max_depth=5,n_estimators=100).fit(X_train_df, y_train_df)
rf.score(X_train_df, y_train_df) 


# In[39]:

y_pred = rf.predict(X_test_df)
np.savetxt('Titanic_SN.csv', np.c_[range(892,892+len(X_test_df)),y_pred], delimiter=',', header = 'PassengerId,Survived', comments = '', fmt='%d')


# In[40]:

features = pd.DataFrame()
features['feature'] = X_train_df.columns
features['importance'] = rf.feature_importances_
features.sort(['importance'],ascending=False)


# In[ ]:



