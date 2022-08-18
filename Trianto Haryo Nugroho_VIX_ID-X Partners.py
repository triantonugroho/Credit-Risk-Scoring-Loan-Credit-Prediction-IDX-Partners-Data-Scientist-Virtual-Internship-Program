#!/usr/bin/env python
# coding: utf-8

# ![Video+Learning+1_page-0001.jpg](attachment:Video+Learning+1_page-0001.jpg)

# # Final Project Data Scientist Virtual Internship Experience ID/X Partners

# ## Credit Risk Scoring Prediction

# By : Trianto Haryo Nugroho

# ### Description :
# 
# This is the final project of my contract period as an Intern Data Scientist at ID/X Partners. I was involved in the project of a lending company. I've collaborate with various other departments on this project to provide technology solutions for the company. I built a model that can predict credit risk using a company-provided data set consisting of accepted and rejected request data. In addition, I also provide visual media to present solutions to clients. The visual media you create is clear, easy to read, and communicative. The work on this end-to-end solution is carried out in the Python programming language while still referring to the Data Science framework/methodology.

# ### Use Case
# 
# Credit Risk Scoring Prediction
# 

# #### Objective Statement:
# * Get business insight about the distribution of the 4 credit scoring classifications
# * Reducing the risk in deciding to apply for a credit loan that has a bad credit score
# * Increase income by accepting credit loan applications with a good credit score

# #### Challenges:
# * Large size of data that is still not clean and various data types
# * Need several coordination with various other department
# * Original dataset from clients who need immediate business problem solutions

# #### Methodology / Analytic Technique:
# * Descriptive Analysis
# * Diagnostic Analysis
# * Predictive Analysis
# * Multiclass Classification Algorithm

# #### Business Benefit:
# * Helping Risk Management Team to create credit score prediction for credit loan application
# * Knowing the factors that affect the credit score

# #### Expected Outcome:
# * Knowing the factors that affect the credit score
# * Get a credit loan prediction model with the best accuracy
# * Web-based credit score prediction application

# ### Business Understanding
# * Credit scoring is a statistical analysis performed by lenders and financial institutions to determine the creditworthiness of a person or a small, owner-operated business. Credit scoring is used by lenders to help decide whether to extend or deny credit.
# * What are the factors that cause a good credit score?
# * What are the factors that cause a bad credit score?
# * What kind of prediction model is the best for credit score prediction?
# * What recommendations are given to lending companies to accept or reject a credit loan?

# ### Data Understanding
# * Data is loan credit historical dataset from 2007 - 2014
# * The dataset consists of 466,285 rows and 75 columns

# #### Data Dictionary:
# 1. **unnamed** : Index. 
# 2. **id** : A unique loan credit assigned ID for the loan listing.
# 3. **member_id** : A unique LC assigned ID for the borrower.
# 4. **loan_amnt** : Last month payment was received.
# 5. **funded_amnt** : The total amount committed to that loan at that point in time.
# 6. **funded_amnt_inv** : The total amount committed to that loan at that point in time for portion of total amount funded by investors.
# 7. **term** : The number of payments on the loan. Values are in months and can be either 36 or 60.
# 8. **int_rate** : Interest Rate on the loan.
# 9. **installment** : The monthly payment owed by the borrower if the loan originates.
# 10. **grade** : LC assigned loan grade.
# 11. **sub_grade** : LC assigned loan grade.
# 12. **emp_title** : The job title supplied by the Borrower when applying for the loan.
# 13. **emp_length** : Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
# 14. **home_ownership** : The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.
# 15. **annual_inc** : The self-reported annual income provided by the borrower during registration.
# 16. **verification_status** : Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified.
# 17. **issue_d** : The month which the loan was funded.
# 18. **loan_status** : Current status of the loan.
# 19. **pymnt_plan** : Indicates if a payment plan has been put in place for the loan.
# 20. **url** : URL for the LC page with listing data.
# 21. **desc** : Loan description provided by the borrower.
# 22. **purpose** : A category provided by the borrower for the loan request.
# 23. **title** : The loan title provided by the borrower.
# 24. **zip_code** : The first 3 numbers of the zip code provided by the borrower in the loan application.
# 25. **addr_state** : The state provided by the borrower in the loan application.
# 26. **dti** : A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
# 27. **delinq_2yrs** : The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years.
# 28. **earliest_cr_line** : The month the borrower's earliest reported credit line was opened.
# 29. **inq_last_6mths** : The number of inquiries in past 6 months (excluding auto and mortgage inquiries).
# 30. **mths_since_last_delinq** : The number of months since the borrower's last delinquency.
# 31. **mths_since_last_record** : The number of months since the last public record.
# 32. **open_acc** : The number of open credit lines in the borrower's credit file.
# 33. **pub_rec** : Number of derogatory public records.
# 34. **revol_bal** : Total credit revolving balance.
# 35. **revol_util** : Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.        
# 36. **total_acc** : The total number of credit lines currently in the borrower's credit file.
# 37. **initial_list_status** : The initial listing status of the loan. Possible values are – W, F.
# 38. **out_prncp** : Remaining outstanding principal for total amount funded.
# 39. **out_prncp_inv** : Remaining outstanding principal for portion of total amount funded by investors.
# 40. **total_pymnt** : Payments received to date for total amount funded.
# 41. **total_pymnt_inv** : Payments received to date for portion of total amount funded by investors.
# 42. **total_rec_prncp** : Principal received to date.
# 43. **total_rec_int** : Interest received to date.
# 44. **total_rec_late_fee** : Late fees received to date.
# 45. **recoveries** : Post charge off gross recovery.
# 46. **collection_recovery_fee** : Post charge off collection fee.
# 47. **last_pymnt_d** : Last month payment was received.
# 48. **last_pymnt_amnt** : Last total payment amount received.
# 49. **next_pymnt_d** : Last month payment was received.
# 50. **last_credit_pull_d** : The most recent month LC pulled credit for this loan.
# 51. **collections_12_mths_ex_med** : Number of collections in 12 months excluding medical collections.
# 52. **mths_since_last_major_derog** : Months since most recent 90-day or worse rating.
# 53. **policy_code** : publicly available policy_code=1, new products not publicly available policy_code=2.
# 54. **application_type** : Indicates whether the loan is an individual application or a joint application with two co-borrowers.
# 55. **annual_inc_joint** : The combined self-reported annual income provided by the co-borrowers during registration.
# 56. **dti_joint** : A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income.
# 57. **verification_status_joint** : Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified.
# 58. **acc_now_delinq** : The number of accounts on which the borrower is now delinquent.
# 59. **tot_coll_amt** : Total collection amounts ever owed.
# 60. **tot_cur_bal** : Total current balance of all accounts.
# 61. **open_acc_6m** : Number of open trades in last 6 months.
# 62. **open_il_6m** : Number of currently active installmant trades.
# 63. **open_il_12m** : Number of installment accounts opened in past 12 months.
# 64. **open_il_24m** : Number of installment accounts opened in past 24 months.
# 65. **mths_since_rcnt_il** : Months since most recent installment accounts opened.
# 66. **total_bal_il** : Total current balance of all installment accounts.
# 67. **il_util** : Ratio of total current balance to high credit/credit limit on all install acct.
# 68. **open_rv_12m** : Number of revolving trades opened in past 12 months.
# 69. **open_rv_24m** : Number of revolving trades opened in past 24 months.
# 70. **max_bal_bc** : Maximum current balance owed on all revolving accounts.
# 71. **all_util** : Balance to credit limit on all trades.
# 72. **total_rev_hi_lim** : Total revolving high credit/credit limit.
# 73. **inq_fi** : Number of personal finance inquiries.
# 74. **total_cu_tl** : Number of finance trades.
# 75. **inq_last_12m** : Number of credit inquiries in past 12 months. 

# ### Data Preparation
# * Python Version : 3.8.8
# * Packages : Pandas, Numpy, Matplotlib, Seaborn, Sklearn, statsmodels.api, etc

# ### I. Load Library & Data

# In[34]:


# essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import statsmodels.api as sm
import sklearn

# import scikit-learn libray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Scikit-learn Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# import xgboost library
from xgboost import XGBClassifier

# capping outlier library winsorizer
from feature_engine.outliers import Winsorizer,OutlierTrimmer

# removing warnings in cell
import warnings
warnings.filterwarnings('ignore')


# provides version library information to save if this notebook is to be run in the future, which may be some code is out of date.

# In[7]:


print('Pandas Version     :',pd.__version__)
print('Numpy Version      :',np.__version__)
print('Statsmodels Version:',sm.__version__)
print('Matplotlib Version :',matplotlib.__version__)
print('Seaborn Version    :',sns.__version__)
print('Sklearn Version    :',sklearn.__version__)


# open dataset

# In[8]:


df_ori = pd.read_csv('loan_data_2007_2014.csv', low_memory=False)


# looking at the dataset information, there are 466285 rows and 75 columns

# In[9]:


df_ori.info()


# view information from a column of type object data

# In[10]:


df_ori.select_dtypes(include=['object']).describe().T


# urls have high cardinality because no rows are the same. the rest nothing too significant can be taken to be informative.

# In[11]:


df_ori.select_dtypes(include=np.number).describe().T


# view column data type numeric. there are a lot of empty columns, this column is useless and can be dropped directly

# In[12]:


df = df_ori.dropna(axis=1,how='all')


# see the contents of the object or categorical column to understand what the content of the category of each column is

# In[13]:


for i in df.select_dtypes(include=['object']):
    print(i)
    print(df[i].unique())
    print("-"*50)


# ### II. Exploratory Data Analysis
# 

# first take 10 samples which will be used as inference data, or try it when the model has been completed

# In[14]:


data_inf = df.sample(10,random_state=25)
data_inf


# dropping the inference data from the dataset then resetting the index of the dataset as well as the inference data

# In[15]:


data = df.drop(data_inf.index, axis=0)
data.reset_index(drop=True, inplace=True)
data_inf.reset_index(drop=True, inplace=True)
print(data.shape)
print(data_inf.shape)


# see the distribution of the target data label. this time the loan_status kolom column

# In[16]:


print(data['loan_status'].value_counts())
print('-'*50)
print('percentage loan_status : \n',round((data['loan_status'].value_counts()/len(data))*100,2))
print('-'*50)
print('percentage null: \n',(data['loan_status'].isnull().sum()/len(data))*100)


# there are 8 categories in loan_status and there is no missing value. try to do the visualization of the column

# In[17]:


plt.figure(figsize=(10,5))
sns.countplot(y='loan_status',data=data,palette='Blues_r')
plt.show()


# because there are many target label classifications, it will be simplified to Excellent, good, poor, bad.
# 
# * excellent means that the person has a credit history that has been fully paid and has no problems
# * good means that the person is doing credit and has never had a problem
# * poor means that the person has a history of late payments
# * Bad means that the person has a history of default

# In[18]:


data['loan_status'] = data['loan_status'].replace(
    {'Fully Paid':'excellent',
    'Current':'good',
    'Charged Off': 'bad',
    'Default':'bad',
    'In Grace Period':'poor',
    'Late (16-30 days)':'poor',
    'Late (31-120 days)':'poor',
    'Does not meet the credit policy. Status:Charged Off':'bad',
    'Does not meet the credit policy. Status:Fully Paid':'bad'})


# when visualized it will be like this

# In[19]:


plt.figure(figsize=(10,5))
sns.countplot(y='loan_status',data=data, palette='Blues_r',order=['excellent','good','poor','bad'])
plt.savefig('picture/loanstat.png')
plt.title('Target Label')
plt.show()


# the target label looks imbalanced, but it's not too bad, so later when splitting the data, stratification is done to divide the classification according to the ratio

# ### Target Label Analysis

# In[20]:


plt.figure(figsize=(10,5))
sns.countplot(data=data, y='loan_status', hue='grade', order=['excellent','good','poor','bad'],
                hue_order = ['A','B','C','D','E','F','G'],palette='Blues_r')
plt.savefig('picture/grade_target.png')
plt.title('Grade & Target Label')
plt.show()


# When compared to loan status, there are some with Grade A that are still classified as poor and bad, and vice versa. There are several Grade G which are classified into good and excellent categories. Grading does not necessarily indicate that the credit rating will be good too.

# In[21]:


plt.figure(figsize=(10,5))
sns.countplot(data=data, y='loan_status', hue='verification_status', order=['excellent','good','poor','bad'], palette='Blues_r')
plt.savefig('picture/verified_target.png')
plt.title('Verivication Status & Target Label')
plt.show()


# When compared to verification status, even though there are many borrowers who do not have Verified status, it does not guarantee that the Credit Rating will be good either.

# In[23]:


plt.figure(figsize=(15,7))
sns.countplot(data=data, x='loan_status', hue='purpose', order=['excellent','good','poor','bad'], palette='Set2',
            hue_order=data['purpose'].value_counts().index)
plt.savefig('picture/purpose_target.png')
plt.title('Loan Purpose & Target Label')
plt.show()


# The purpose of most borrowing is debt consolidation and then credit cards. The distribution of loan objectives and each classification of target labels are quite similar, so that no single loan objective is categorized as bad credit risk.

# In[24]:


plt.figure(figsize=(20,8))
abs_values = df['purpose'].value_counts(ascending=False).values
ax = sns.countplot(x='purpose',data=data, order=data['purpose'].value_counts().index, palette='Set2')
ax.bar_label(container=ax.containers[0], labels=abs_values, padding=3)
plt.savefig('picture/purpose.png')
plt.title('Loan Purpose')
plt.show()


# ### III. Data Splitting

# splitting data using stratify, and test data by 20% of the dataset.

# In[25]:


x = data.drop(['loan_status'], axis=1)
y = data['loan_status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25,stratify=y)
print('xtrain dataset shape:',x_train.shape)
print('xtest dataset shape:',x_test.shape)
print('ytrain dataset shape:',y_train.shape)
print('ytest dataset shape:',y_test.shape)


# ### IV. Outlier Handling

# In carrying out outlier handling, there are several stages that researchers do to get clean data. the summary of the data cleaning stages is as follows:
# 
# * create a treatment table, this is used to find out the distribution, lcl, ucl and treatments that can be done to the train data for outlier handling.
# * perform outlier handling on normal data and must be capped by means of a winorizer
# * perform outlier handling on data that must be trimmed with outlier trimmer
# 
# for more details, it will be done the same as the steps above
# 

# In[27]:


# create a function for skewed data and search for the lower and upper limits using IQR
def iqr(data,column):
    lower_limit= data[column].quantile(0.25) - 3*(data[column].quantile(0.75)-data[column].quantile(0.25))
    upper_limit= data[column].quantile(0.75) + 3*(data[column].quantile(0.75)-data[column].quantile(0.25))
    return lower_limit, upper_limit

# create a function for normally distributed data to find the upper and lower limits using the standard deviation.
def lcl_ucl_std(data,column):
    lcl = data[column].mean() - 1.5*data[column].std()
    ucl = data[column].mean() + 1.5*data[column].std()
    return lcl,ucl

# create a function to find out how many percentages of outliers for each of the skewed and normal distributions.
# if there are no outliers from the column then the value will be 0

def perc_outliers_iqr(data,column):
    lcl,ucl = iqr(data,column)
    try:
        result = data[column][(data[column]<=lcl) | (data[column]>=ucl)].count()/len(data[column])
    except:
        result = 0
    return result

def perc_outliers_std(data,column):
    lcl,ucl = lcl_ucl_std(data,column)
    try:
        result = data[column][(data[column]<=lcl) | (data[column]>=ucl)].count()/len(data[column])
    except:
        result = 0
    return result
    
# create a function to concatenate the IQR and std deviation functions. with the selection of functions seen from the skewness of the data.
# if above -0.5 and below 0.5 then the data is normally distributed.
# while outside of this value, the data has a skew distribution
def outliers(data,column,distr):
    if distr <= -0.5 or distr >= 0.5:
        lcl,ucl = iqr(data,column)
        percentage = perc_outliers_iqr(data,column)
    elif distr > -0.5 or distr < 0.5:
        lcl,ucl = lcl_ucl_std(data,column)
        percentage = perc_outliers_std(data,column)
    return lcl,ucl,percentage


# To make it easier for which columns are skewed or normal and what treatment should be done, I created a treatment table whose contents are numeric columns with the following descriptions:

# Column Name       : Description
# 
# * name 	          : numeric column name
# 
# * distr           : distribution skew value
# 
# * percentage	  : the percentage of the number of outliers outside the lower and upper limits
# 
# * skewness        : distribution name
# 
# * treatment	      : treatment that must be done
# 

# In[28]:


num_columns = x_train.select_dtypes(include=np.number).columns


# In[30]:


# add skew column data to x_train numerik numeric column
skew = []
for i in x_train[num_columns]:
    skew.append(x_train[i].skew())
# add data to the percentage column by using the outliers . function
percentage = []
j = 0
for i in x_train[num_columns]:
    percentage.append(outliers(x_train,i,skew[j])[2])
    j += 1

# add a treatment column to find out what treatment should be done to the feature
treatment = []
j=0
for i in x_train[num_columns]:
    if percentage[j] == 0:
        treatment.append('No outliers')
    elif percentage[j] <= 0.05:
        treatment.append('trim')
    elif percentage[j] <= 0.15:
        treatment.append('capping')
    elif percentage[j] > 0.15:
        treatment.append('do not treat')
    j += 1

# add the distribution column by looking at the skewed value.
# if the value is -0.5 to 0.5 then it is normally distributed. outside this value, the distribution is skewed
distribution = []
j = 0
for i in x_train[num_columns]:
    if skew[j] >= -0.5 and skew[j] <= 0.5:
        distribution.append('normal')
    else: 
        distribution.append('skewed')
    j += 1

# create a dataframe with the outlier_treatment variable to be the treatment table frame
outlier_treatment = pd.DataFrame()
outlier_treatment['name'] = x_train[num_columns].columns
outlier_treatment['distr'] = distribution
outlier_treatment['percentage'] = percentage
outlier_treatment['skewness'] = skew
outlier_treatment['treatment'] = treatment


# looking at the outlier treatment table, from here we can perform treatment on features according to their distribution and the percentage of the outliers.

# In[31]:


outlier_treatment


# Capping features are normally distributed

# In[32]:


windsoriser = Winsorizer(capping_method='gaussian', # choose gaussian for mean and std
                          tail='both', # cap both tails 
                          fold=1.5,
                          variables=outlier_treatment[(outlier_treatment['treatment'] == 'capping') & (outlier_treatment['distr'] == 'normal')]['name'].tolist(),
                            missing_values='ignore')

windsoriser.fit(x_train)
x_train_cap_norm = windsoriser.transform(x_train)


# The capping feature has a skewed distribution

# In[33]:


windsoriser = Winsorizer(capping_method='iqr', # choose iqr for mean and std
                          tail='both', # cap both tails 
                          fold=1.5,
                          variables=outlier_treatment[(outlier_treatment['treatment'] == 'capping') & (outlier_treatment['distr'] == 'skewed')]['name'].tolist(),
                            missing_values='ignore')

windsoriser.fit(x_train_cap_norm)
x_train_cap_skew = windsoriser.transform(x_train_cap_norm)


# before trimming, x_train is combined with y_train so that the number of rows of the dataset will remain the same

# In[35]:


y_train_trim = y_train.copy()
x_train_cap_skew = pd.concat([x_train_cap_skew,y_train_trim],axis=1)


# trim features are normally distributed

# In[36]:


trimmer = OutlierTrimmer(capping_method='gaussian', # choose gaussian for mean and std
                        tail= 'both',
                        fold=3,
                        variables=outlier_treatment[(outlier_treatment['treatment'] == 'trim') & (outlier_treatment['distr'] == 'normal')]['name'].tolist(),
                        missing_values='ignore')

x_train_trim_norm = trimmer.fit(x_train_cap_skew)
x_train_trim_norm = trimmer.transform(x_train_cap_skew)


# trim feature with skewed distribution

# In[37]:


trimmer = OutlierTrimmer(capping_method='iqr', # choose iqr for mean and std
                        tail= 'both',
                        fold=3,
                        variables=outlier_treatment[(outlier_treatment['treatment'] == 'trim') & (outlier_treatment['distr'] == 'skewed')]['name'].tolist(),
                        missing_values='ignore')

x_train_outlier_clean = trimmer.fit(x_train_trim_norm)
x_train_outlier_clean = trimmer.transform(x_train_trim_norm)


# data has been done outlier handling

# In[38]:


y_train_outlier_clean = x_train_outlier_clean['loan_status']
x_train_outlier_clean = x_train_outlier_clean.drop(['loan_status'],axis=1)

print('Before Outlier handling:', x_train.shape)
print('After Outlier handling:', x_train_outlier_clean.shape)
print(f'Ratio Outlier Handing : {x_train_outlier_clean.shape[0]/x_train.shape[0]:.2%}')


# the conclusion is there are still 93.36% of data that can still be used for machine learning after outlier handling

# ### V. Missing Value Handling

# First of all, we will do missing value handling on features that are less than 6% missing

# In[39]:


percentage_null = x_train_outlier_clean.isnull().sum()/x_train_outlier_clean.shape[0]
round(percentage_null[(percentage_null <=0.06)&(percentage_null>0)],4)*100


# In[40]:


emp_length is the length of the job in years. Possible values ​​are between 0 and 10 where 0 means less than one year and 10 means ten years or more.


# which means it is a categorical feature. If you look at the distribution as follows:

# In[41]:


sns.countplot(y = x_train_outlier_clean['emp_length'])


# because the feature is categorical, the value that most often appears is 10+ years

# In[42]:


impute_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

emp_length_impute = impute_frequent.fit_transform(x_train_outlier_clean[['emp_length']])
x_train_outlier_clean['emp_length'] = emp_length_impute


# for emp_title it means the Job Title provided by the borrower when applying for a loan.
# 
# then it can also be imputed with the value that occurs most often

# In[43]:


emp_title_impute = impute_frequent.fit_transform(x_train_outlier_clean[['emp_title']])
x_train_outlier_clean['emp_title'] = emp_title_impute


# then trimming for features that have a very small percentage of missing values

# In[44]:


x_train_MV_trim = pd.concat([x_train_outlier_clean,y_train_outlier_clean],axis=1)


# In[45]:


x_train_MV_trim = x_train_MV_trim.dropna(subset=['annual_inc', 'title', 'delinq_2yrs', 'earliest_cr_line',
                                        'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_util', 'total_acc',
                                        'last_pymnt_d', 'last_credit_pull_d', 'collections_12_mths_ex_med',
                                        'acc_now_delinq'])


# then handling with a larger percentage missing value

# In[46]:


round(percentage_null[(percentage_null >0.06)&(percentage_null>0)],4)*100


# In[47]:


take features that have a missing value of more than 6%


# In[48]:


percentage_null[(percentage_null >0.06)&(percentage_null>0)].index

see the type of data and determine how to handle each feature
# In[49]:


x_train_outlier_clean[['desc', 'mths_since_last_delinq', 'mths_since_last_record',
       'next_pymnt_d', 'mths_since_last_major_derog', 'tot_coll_amt',
       'tot_cur_bal', 'total_rev_hi_lim']]


# In order to simplify the imputation, I will simply imput the median for features with numeric data type, while for object data types, I will imputation with the most frequent imputation.

# In[50]:


impute_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

# fit dan transform setiap kolom
mths_since_last_delinq = impute_mean.fit_transform(x_train_MV_trim[['mths_since_last_delinq']])
mths_since_last_record = impute_mean.fit_transform(x_train_MV_trim[['mths_since_last_record']])
mths_since_last_major_derog = impute_mean.fit_transform(x_train_MV_trim[['mths_since_last_major_derog']])
tot_coll_amt = impute_mean.fit_transform(x_train_MV_trim[['tot_coll_amt']])
tot_cur_bal = impute_mean.fit_transform(x_train_MV_trim[['tot_cur_bal']])
total_rev_hi_lim = impute_mean.fit_transform(x_train_MV_trim[['total_rev_hi_lim']])


# memasukan fit dan transform setiap kolom kembali ke dataset.
x_train_MV_trim['mths_since_last_delinq'] = mths_since_last_delinq
x_train_MV_trim['mths_since_last_record'] = mths_since_last_record
x_train_MV_trim['mths_since_last_major_derog'] = mths_since_last_major_derog
x_train_MV_trim['tot_coll_amt'] = tot_coll_amt
x_train_MV_trim['tot_cur_bal'] = tot_cur_bal
x_train_MV_trim['total_rev_hi_lim'] = total_rev_hi_lim


# in the "desc" column this is a description of the transaction. of these features are not suitable for inclusion in machine learning. then the simplest imputation is not carried out, namely most frequent
# 
# the "next_payment_d" column is filled with backwards fill because it is possible that the data will be inputted according to the input date, so the fill with the closest row will be better than impute with the most frequent
# 

# In[51]:


desc = impute_frequent.fit_transform(x_train_MV_trim[['desc']])
next_payment_d = x_train_MV_trim['next_pymnt_d'].fillna(method='bfill')

x_train_MV_trim['desc'] = desc
x_train_MV_trim['next_pymnt_d'] = next_payment_d


# checking for missing values ​​again after missing value handling

# In[52]:


x_train_MV_trim.isnull().sum()


# data has been completed missing value handling
# 
# dropping the target label loan_status after missing value handling

# In[53]:


x_train_clean = x_train_MV_trim.copy()
y_train_clean = x_train_MV_trim['loan_status']
x_train_clean.drop('loan_status',axis=1,inplace=True)


# In[54]:


print('jumlah data pada clean train data: ',x_train_clean.shape)
print('jumlah data pada train data awal: ',x_train.shape)
print(f'Ratio all outliers and missing value trimming : {x_train_clean.shape[0]/x_train.shape[0]:.2%}')


# summary of missing value handling, data that can still be used is 93.19% of the initial dataset

# ### VI. Feature Selection

# perform feature selection by looking at the correlation of the feature with the target label. heatmap is used to make it easier to retrieve features

# In[55]:


label_enc = LabelEncoder()
y_train_enc = label_enc.fit_transform(y_train_clean)
y_train_enc = pd.Series(y_train_enc,index=y_train_clean.index)

heatmap_after = x_train_clean.copy()
heatmap_after = pd.concat([heatmap_after, y_train_enc], axis=1)

fig = plt.figure(figsize=(30,30))
fig = sns.heatmap(heatmap_after.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
plt.show()


# taken correlation above 0.2 and below -0.2 because there is a correlation directly and inversely with the target. then you get a list of features below

# In[56]:


num_select = ['inq_last_6mths','out_prncp','total_rec_prncp','recoveries','last_pymnt_amnt']


# For category features, encoding is done first with an ordinal encoder, then the correlation is checked using the Spearman method, which is different from the previous one using Pearson. spearman is used for categorical features

# In[57]:


obj_columns = x_train_clean.select_dtypes(include=['object']).columns

# encodinng dengan ordinal encoder
ord_encoder = OrdinalEncoder()
x_train_ord_enc = x_train_clean.copy()
x_train_ord_enc[obj_columns] = ord_encoder.fit_transform(x_train_ord_enc[obj_columns])

# membuat dataframe feature kategori dan target label
heatmap_category = x_train_ord_enc[obj_columns].copy()
heatmap_category = pd.concat([heatmap_category, y_train_enc], axis=1)

# membuat visualisasi heatmap
fig3 = plt.figure(figsize=(30,30))
fig3 = sns.heatmap(heatmap_category.corr(method='spearman'),annot=True,cmap='RdYlGn',linewidths=0.2)
plt.show()


# dengan berpegang dengan cara yang sama dengan sebelumnya maka didapatakan 2 feature. karena feature desc memiliki missing value 73% maka korelasi berbanding terbailik ini tidak bisa dipercaya. bisa dilakukan NLP pada feature ini untuk memprediksi credit score juga. namun pada kali ini tidak akan digunakan, bisa menjadi improvement selanjutnya.

# In[58]:


cat_select = ['initial_list_status','last_pymnt_d']


# ### VII. Modeling

# melakukan feature selection pada dataset x_train dan x_test, pada x_test dilakukan handling missing value saja dengan cara drop.

# In[59]:


# select feature yang akan dimasukan kedalam model dari list yang telah dibuat sebelumnya
x_train_select = x_train_clean[num_select+cat_select]

# memilih feature yang akan dimasukan kedalam model dari list yang telah dibuat sebelumnya
# pada data test. kemudian melakukan treatment missing value
x_test_select = x_test[num_select+cat_select]
x_test_select = pd.concat([x_test_select,y_test],axis=1)

#drop missing value
x_test_select.dropna(inplace=True)

# membagi kembali target label dan x_test
y_test = x_test_select['loan_status']
x_test_select.drop('loan_status',axis=1,inplace=True)


# In[60]:


x_train_select.head()


# Doing modeling using pipelines. The scaler used is a standard scaler, and the encoding used is an ordinal encoder. because the column 'initial_list_status' has 2 categories and 'last_pymnt_d' is a date then an ordinal encoder is used

# In[61]:


transformer_std = StandardScaler()
transformer_ord = OrdinalEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', transformer_std, num_select),
        ('cat', transformer_ord, cat_select),
    ])

clf_decision_tree = Pipeline(steps=[('preprocessor', preprocessor),('classifier', DecisionTreeClassifier(random_state=42))])
clf_random_forest = Pipeline(steps=[('preprocessor', preprocessor),('classifier', RandomForestClassifier(random_state=42))])
clf_knn = Pipeline(steps=[('preprocessor', preprocessor),('classifier', KNeighborsClassifier())])
clf_gaussian_nb = Pipeline(steps=[('preprocessor', preprocessor),('classifier', GaussianNB())])
clf_xgb = Pipeline(steps=[('preprocessor', preprocessor),('classifier', XGBClassifier(random_state=42,verbosity=0))])


# the models to be used are decision tree, random forest, K nearest neighbor, naive bayes, and xtreme gradient boosting with default hyperparameters
# 
# then training with each model

# In[62]:


clf_decision_tree.fit(x_train_clean, y_train_clean)
clf_random_forest.fit(x_train_clean, y_train_clean)
clf_knn.fit(x_train_clean, y_train_clean)
clf_gaussian_nb.fit(x_train_clean, y_train_clean)
clf_xgb.fit(x_train_clean, y_train_enc)


# scoring for each model

# In[63]:


score_decision_tree = clf_decision_tree.score(x_train_select, y_train_clean)
score_random_forest = clf_random_forest.score(x_train_select, y_train_clean)
score_knn = clf_knn.score(x_train_select, y_train_clean)
score_gaussian_nb = clf_gaussian_nb.score(x_train_select, y_train_clean)
score_xgb = clf_xgb.score(x_train_select, y_train_enc)


# In[64]:


print(f'Score Decision Tree : {score_decision_tree:.2%}')
print(f'Score Random Forest : {score_random_forest:.2%}')
print(f'Score KNN : {score_knn:.2%}')
print(f'Score Gaussian NB : {score_gaussian_nb:.2%}')
print(f'Score xgboost : {score_xgb:.2%}')


# obtained decision tree and random forest with almost 100% accuracy, but this must be done cross validation so that the value becomes valid

# ### Cross Validation
# 

# In[67]:


crossval_dt = cross_val_score(clf_decision_tree, x_train_select, y_train_clean, cv=5)
crossval_rf = cross_val_score(clf_random_forest, x_train_select, y_train_clean, cv=5)
crossval_knn = cross_val_score(clf_knn, x_train_select, y_train_clean, cv=5)
crossval_gnb = cross_val_score(clf_gaussian_nb, x_train_select, y_train_clean, cv=5)
crossval_xgb = cross_val_score(clf_xgb, x_train_select, y_train_enc, cv=5)


# crossvalidation with cv = 5
# 
# The results will be seen in the next chapter Evaluation

# ### VIII. Evaluation

# ### Decision Tree Evaluation

# In[68]:


ypred_dt_train = clf_decision_tree.predict(x_train_select)
ypred_dt_test = clf_decision_tree.predict(x_test_select)

print('Accuracy - Test Set   : ', accuracy_score(y_test, ypred_dt_test))
print('Accuracy - Train Set  : ', accuracy_score(y_train_clean, ypred_dt_train))

print('Cross Validation decision tree ---------------------')
print('Accuracy - All - Cross Validation  : ', crossval_dt)
print('Accuracy - Mean - Cross Validation : ', crossval_dt.mean())

print('Classification Report test: \n', classification_report(y_test, ypred_dt_test), '\n')
print('Classification Report train: \n', classification_report(y_train_clean, ypred_dt_train), '\n')


# the model tends to overfit the train dataset. with the test dataset only 96% accuracy. Judging from the F1 score for the "poor" classification, it gets the worst score because there is not much data for the poor classification. for cross validation the average value is 96% accuracy

# ### Random Forest Evaluation

# In[69]:


ypred_rf_train = clf_random_forest.predict(x_train_select)
ypred_rf_test = clf_random_forest.predict(x_test_select)

print('Accuracy - Test Set   : ', accuracy_score(y_test, ypred_rf_test))
print('Accuracy - Train Set  : ', accuracy_score(y_train_clean, ypred_rf_train))

print('Cross Validation random forest ---------------------')
print('Accuracy - All - Cross Validation  : ', crossval_rf)
print('Accuracy - Mean - Cross Validation : ', crossval_rf.mean())

print('Classification Report test: \n', classification_report(y_test, ypred_rf_test), '\n')
print('Classification Report train: \n', classification_report(y_train_clean, ypred_rf_train), '\n')


# the random forest model tends to overfit the train data, the average cross validation value shows 97% accuracy with the f1 score "poor" better than the decision tree

# ### Evaluation of K Nearest Neighbor

# In[71]:


ypred_knn_train = clf_knn.predict(x_train_select)
ypred_knn_test = clf_knn.predict(x_test_select)

print('Accuracy - Test Set   : ', accuracy_score(y_test, ypred_knn_test))
print('Accuracy - Train Set  : ', accuracy_score(y_train_clean, ypred_knn_train))

print('Cross Validation knn---------------------')
print('Accuracy - All - Cross Validation  : ', crossval_knn)
print('Accuracy - Mean - Cross Validation : ', crossval_knn.mean())

print('Classification Report test: \n', classification_report(y_test, ypred_knn_test), '\n')
print('Classification Report train: \n', classification_report(y_train_clean, ypred_knn_train), '\n')


# KNN model has the most consistent accuracy and F1 score and does not overfit. with an average validation score of 96%

# ### Naive Bayes Evaluation

# In[72]:


ypred_nb_train = clf_gaussian_nb.predict(x_train_select)
ypred_nb_test = clf_gaussian_nb.predict(x_test_select)

print('Accuracy - Test Set   : ', accuracy_score(y_test, ypred_nb_test))
print('Accuracy - Train Set  : ', accuracy_score(y_train_clean, ypred_nb_train))

print('Cross Validation naive bayes ---------------------')
print('Accuracy - All - Cross Validation  : ', crossval_gnb)
print('Accuracy - Mean - Cross Validation : ', crossval_gnb.mean())

print('Classification Report test: \n', classification_report(y_test, ypred_nb_test), '\n')
print('Classification Report train: \n', classification_report(y_train_clean, ypred_nb_train), '\n')


# Naive Bayes is not overfit, but for F1 the "bad" score is only around 70%, which affects accuracy. when viewed from cross validation, the average value of accuracy is 94%

# ### XGBoost Evaluation
# 

# before predicting with xgboost, the target label is transformed first, because the xgboost library cannot predict directly from the target label object, it must be numeric

# In[73]:


y_test_enc = label_enc.transform(y_test)


# In[74]:


ypred_xgb_train = clf_xgb.predict(x_train_select)
ypred_xgb_test = clf_xgb.predict(x_test_select)

print('Accuracy - Test Set   : ', accuracy_score(y_test_enc, ypred_xgb_test))
print('Accuracy - Train Set  : ', accuracy_score(y_train_enc, ypred_xgb_train))

print('Cross Validation xgboost---------------------')
print('Accuracy - All - Cross Validation  : ', crossval_xgb)
print('Accuracy - Mean - Cross Validation : ', crossval_xgb.mean())

print('Classification Report test: \n', classification_report(y_test_enc, ypred_xgb_test), '\n')
print('Classification Report train: \n', classification_report(y_train_enc, ypred_xgb_train), '\n')


# for 0 means bad, 1 means excellent, 2 means good and 3 means poor
# 
# the XGBoost model has the best "poor" f1 score of 69%, then when viewed from cross validation the highest accuracy value is 98%. this value is quite large than the other models, and is very good. Therefore, hyperparameter tuning is not performed for this model. because it's been very good.
# 
# XGBoost model is the model that will be used for data inference and deployment.

# In[75]:


# in the comments so the model doesn't fall

# import pickle
# with open('model/model_credit_score.pkl','wb') as f:
# pickle.dump(clf_xgb,f)


# then the model is saved to model_credit_score.pkl

# ### IX. Data Inference

# make predictions on the inference data to try the XGBoost model that has been made

# In[77]:


data_inf


# cleaning target label menjadi yang sesuai dari model gunakan

# In[79]:


data_inf['loan_status'] = data_inf['loan_status'].replace(
    {'Fully Paid':'excellent',
    'Current':'good',
    'Charged Off': 'bad',
    'Default':'bad',
    'In Grace Period':'poor',
    'Late (16-30 days)':'poor',
    'Late (31-120 days)':'poor',
    'Does not meet the credit policy. Status:Charged Off':'bad',
    'Does not meet the credit policy. Status:Fully Paid':'bad'})


# choose the features used for the model

# In[80]:


data_inf_select = data_inf[num_select + cat_select]


# check if there is a missing value

# In[81]:


data_inf_select.isnull().sum()


# Predict and inverse transformation are carried out on the predicted value

# In[82]:


pred_inf = clf_xgb.predict(data_inf_select)
pred_inf = label_enc.inverse_transform(pred_inf)


# see from the target label data inference with prediction results

# In[83]:


data_result = pd.DataFrame()
data_result['actual'] = data_inf['loan_status']
data_result['prediction'] = pred_inf
data_result


# the result is 100% the model can predict the inference data.
# 

# ### X. Summary

# for the target label with grade, verification status and loan purpose, each of which is not very related to the target label. because for the distribution of grade, verification status and loan purpose, each has a similar distribution for each classification of the target label.
# 
# The results of each model of accuracy cross validation are as follows
# 
# * Decision Tree with 96% accuracy
# * Random Forest with 97% accuracy
# * KNN with 96% accuracy
# * Naive Bayes with 94% accuracy
# * XGBoost with 98% accuracy
# 
# So the best model is the XGBoost model so that the model is chosen as the model for deployment
# 
# the results of this model can be used to help a credit risk analyst to determine whether a borrower can borrow with the least risk by lenders. By using a model that has an accuracy of 98%, a credit risk analyst can find it easier and faster to determine whether a borrower can get a loan or not from a lender company.
