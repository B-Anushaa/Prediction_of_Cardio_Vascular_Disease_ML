
import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
import pickle
Data Analysys
data1=pd.read_csv("/content/archive (6) (1).zip")
data1.head()
data1.shape
data1.info()
data1.isnull().sum()
data1.describe()
data1['target'].value_counts()
plt.title('HEART DISEASE DISTRIBUTION')
data1['target'].value_counts().plot(kind="pie",autopct='%.1f%%', figsize=(8,8),shadow=True)
plt.title('Age Vs Heart Disease')
sns.boxplot(x='target', y='age', data=data1,palette='rainbow')
data1.groupby(['sex'])['target'].value_counts()
sns.countplot(x=data1['sex'],hue=data1['target'])
plt.show()
sns.boxplot(x='target', y='thalach', data=data1,palette='rainbow')
sns.stripplot(x="target", y="thalach", data=data1)
sns.scatterplot(data=data1, x="age", y="thalach", hue="target")
data1.groupby(['exang'])['target'].value_counts()
sns.countplot(x=data1['exang'],hue=data1['target'])
plt.show()
sns.boxplot(x='target', y='oldpeak', data=data1,palette='rainbow')
data1.groupby(['slope'])['target'].value_counts()
sns.countplot(x=data1['slope'],hue=data1['target'])
plt.show()
data1.groupby(['ca'])['target'].value_counts()
sns.countplot(x=data1['ca'],hue=data1['target'])
plt.show()
sns.countplot(x=data1['thal'],hue=data1['target'])
plt.show()
plt.figure(figsize = (15,15))
sns.heatmap(data1.corr(), vmin = -1, vmax = +1, annot = True, cmap = 'coolwarm')
X = data1.drop(['target'],axis='columns')
xix
X.head(10)
y = data1.target
y.head(3)
len(x)
len(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
model_1 = MultinomialNB()
model_1.fit(X_train, y_train)
nb=model_1.score(X_test, y_test)
nb
Logistic Regression
model_2 = LogisticRegression()
model_2.fit(X_train, y_train)
lr=model_2.score(X_test, y_test)
lr
Random Forest
model_3 = RandomForestClassifier(n_estimators=30)
model_3.fit(X_train, y_train)
rf=model_3.score(X_test, y_test)
rf
Decision Tree
model_4 = tree.DecisionTreeClassifier(criterion='entropy')
model_4.fit(X_train, y_train)
dt=model_4.score(X_train, y_train)
dt
Support Vector Machine(SVC)
model_5 = SVC()
model_5.fit(X_train, y_train)
sv=model_5.score(X_test, y_test)
sv
accuracy = [nb,lr,rf,dt,sv]
all_models =
['NaiveBayesClassifier','LogisticRegression','RandomForestClassifier','DecisonTreeClassifier'
,'SVC']
score_df = pd.DataFrame({'Algorithms': all_models, 'Accuracy_Score': accuracy})
score_df.style.background_gradient(cmap="YlGnBu",high=1,axis=0)
mylist=[]
mylist2=[]
mylist.append(nb)
mylist2.append("Naive Bayes")
mylist.append(lr)
mylist2.append("Logistic Regression")
mylist.append(rf)
mylist2.append("Random Forest")
mylist.append(dt)
mylist2.append("Decision Tree")
mylist.append(sv)
mylist2.append("SVM")
plt.rcParams['figure.figsize']=8,6
sns.set_style("darkgrid")
pal_style=['#F38BB2','#4C0028','#8A0030','#100C07','#FF0000']
ax = sns.barplot(x=mylist2, y=mylist, palette = pal_style, saturation =1.5)
plt.xlabel("Classification Models", fontsize = 20 )
plt.ylabel("Accuracy", fontsize = 20)
plt.title("Accuracy of different Classification Models", fontsize = 20)
plt.xticks(fontsize = 11, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
for p in ax.patches:
width, height = p.get_width(), p.get_height()
x, y = p.get_xy()
ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()
model_4.predict([[56,1,1,120,236,0,1,178,0,0.8,2,0,2]])
model_4.predict([[57,1,0,130,131,0,1,115,1,1.2,1,1,3]])
