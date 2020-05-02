import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt	# for ploting of graph
import seaborn as sns		# for ploting of graph
from sklearn.impute import KNNImputer	# for KNN Imputing missing values
from scipy.stats import chi2_contingency	# for chi square test
from sklearn import tree			# for decission tree
from sklearn.metrics import *			# for decission tree
from sklearn.model_selection import train_test_split	# for splitting the data 
from sklearn.ensemble import RandomForestClassifier	# for random forrest
import statsmodels.api as sm	# for logistic regression
from sklearn.neighbors import KNeighborsClassifier	# for KNN machine learning algorithm
from sklearn.naive_bayes import GaussianNB		# for naive_bayes 

os.chdir("F:/proj2")

#read data
raw_dat=pd.read_csv("bank-loan.csv")

#convert the numerical variables into required categorical variables
raw_dat = raw_dat.astype({"ed":'category', "default":'category'}) 

#plot boxplot to visualize outliers
sns.boxplot(raw_dat['age'])
plt.boxplot(raw_dat['employ'])
plt.boxplot(raw_dat['address'])
sns.boxplot(raw_dat['income'])
plt.boxplot(raw_dat['debtinc'])
plt.boxplot(raw_dat['CASH_ADVANCE_TRX'])
plt.boxplot(raw_dat['creddebt'])
sns.boxplot(raw_dat['othdebt'])


#continous variables
c=['age','employ','address','income','debtinc','creddebt','othdebt']


#finding and removing outliers
for i in c:
    q75,q25=np.nanpercentile(raw_dat.loc[:,i],[75,25])
    iqr=q75-q25
    min=q25-(iqr*1.5)
    max=q75+(iqr*1.5)
    print(i)
    print(min)
    print(max)
    raw_dat.loc[raw_dat[i]<min,:i] =np.nan
    raw_dat.loc[raw_dat[i]>max,:i] =np.nan

#store column names
cols=list(raw_dat)

#impute with KNN
KNN = KNNImputer(n_neighbors=3)
raw_dat=KNN.fit_transform(raw_dat)

#convert back to data frame from the result of KNN which is array
raw_dat=pd.DataFrame(raw_dat)
raw_dat.columns=cols

#correlation analysis
corr_df=raw_dat.loc[:,c]

#PLotting correlation map
f,ax=plt.subplots(figsize=(7,5))
corr=corr_df.corr()
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)

#chi square test
chi,p,dof,ex=chi2_contingency(pd.crosstab(raw_dat['default'],raw_dat['ed']))
print(p)

#normality check
plt.hist(raw_dat['employ'],bins='auto')


#normalization
for i in c:
    print(i)
    raw_dat[i]=(raw_dat[i]-raw_dat[i].min())/(raw_dat[i].max()-raw_dat[i].min())

#standarization
for i in c:
    print(i)
    raw_dat[i]=(raw_dat[i]-raw_dat[i].mean())/(raw_dat[i].std())


#convert default into yes or no ,to split the target variable into train and test
raw_dat['default']=raw_dat['default'].replace(1,'Yes')

#dicision tree
clf=tree.DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)
y_pred=clf.predict(x_test)
accuracy_score(y_test,y_pred)*100

#confusion matrix
CM=confusion_matrix(y_test,y_pred)
CM=pd.crosstab(y_test,y_pred)
'''
fp=29   23%
fn=26   59%'''

#random forrest
RF_model=RandomForestClassifier(n_estimators=100).fit(x_train, y_train)
rfpreds=RF_model.predict(x_test)
CM=pd.crosstab(y_test,rfpreds)
accuracy_score(y_test,rfpreds)*100

'''
fp=15   11%
fn=35   79%'''

#convert "default" yes or no abck to 0,1,to work with logistic regrression
raw_dat['default']=raw_dat['default'].replace('No',0)
raw_dat['default']=raw_dat['default'].replace('Yes',1)

# logistic regrression
logR=pd.DataFrame(raw_dat['default'])
logR=logR.join(raw_dat[c]) #c- continous variables
SI=np.random.rand(len(logR))<0.8
train=logR[SI]
test=logR[~SI]
tcols=train.columns[1:7]

logit = sm.Logit(train['default'],train[tcols]).fit()
logit.summary()


#predict test data
test['prob']=logit.predict(test[tcols])
test['val']=1
test.loc[test.prob<0.5,'val']=0

CM=pd.crosstab(test['default'],test['val'])
'''
fp=45   36%
fn=15   30%
'''

#KNN

knn=KNeighborsClassifier(n_neighbors=5).fit(x_train,y_train)
knnpred=knn.predict(x_test)
CM=pd.crosstab(y_test,knnpred)
accuracy_score(y_test,knnpred)*100
'''
fp=19   15%
fn=32   72%
'''

#naive bayes
nb=GaussianNB().fit(x_train,y_train)
nbpred=nb.predict(x_test)
CM=pd.crosstab(y_test,nbpred)
accuracy_score(y_test,nbpred)*100
'''
fp=32   25%
fn=30   68%
'''