# Imports
import pandas as pd
import seaborn as sns
sns.set() # Set figure style
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
plt.rcParams['font.family'] = "serif"
from tabulate import tabulate



#### Functions


def model_scoring (model, name, X, y, CV):
  # This function print performance metrics of a classification of 0 and 1 output in a table
  # If metrics for each step of cross validation is of inerest 'CV' should be a model_selection.KFold object
  # If there is no cross-validation the CV should be None

  from sklearn import model_selection 
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import make_scorer

  # defining tn, fp, fn, tp metric functions
  def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
  def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
  def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
  def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
  def tpr(y_true, y_pred):
    tn = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[0, 1]
    fn = confusion_matrix(y_true, y_pred)[1, 0]
    tp = confusion_matrix(y_true, y_pred)[1, 1]
    tpr = tp / (tp + fn)
    return tpr
  def tnr(y_true, y_pred):
    tn = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[0, 1]
    fn = confusion_matrix(y_true, y_pred)[1, 0]
    tp = confusion_matrix(y_true, y_pred)[1, 1]
    tnr = tn / (tn + fp)
    return tnr
  def fpr(y_true, y_pred):
    tn = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[0, 1]
    fn = confusion_matrix(y_true, y_pred)[1, 0]
    tp = confusion_matrix(y_true, y_pred)[1, 1]
    fpr = fp / (tn + fp)
    return fpr
  def fnr(y_true, y_pred):
    tn = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[0, 1]
    fn = confusion_matrix(y_true, y_pred)[1, 0]
    tp = confusion_matrix(y_true, y_pred)[1, 1]
    fnr = fn / (tp + fp)
    return fnr
  def err(y_true, y_pred):
    tn = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[0, 1]
    fn = confusion_matrix(y_true, y_pred)[1, 0]
    tp = confusion_matrix(y_true, y_pred)[1, 1]
    err = (fp + fn) / (tp + fp + fn + tn) # Error rate 
    return err
  def bacc(y_true, y_pred):
    tn = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[0, 1]
    fn = confusion_matrix(y_true, y_pred)[1, 0]
    tp = confusion_matrix(y_true, y_pred)[1, 1]
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    bacc = (tpr + tnr) / 2 # Ballanced Accuracy
    return bacc
  def tss(y_true, y_pred):
    tn = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[0, 1]
    fn = confusion_matrix(y_true, y_pred)[1, 0]
    tp = confusion_matrix(y_true, y_pred)[1, 1]
    tss = (tp / (tp + fn)) - (fp / (fp + tn)) # True Skill Statistics
    return tss
  def hss(y_true, y_pred):
    tn = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[0, 1]
    fn = confusion_matrix(y_true, y_pred)[1, 0]
    tp = confusion_matrix(y_true, y_pred)[1, 1]
    hss = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) # Heidke Skill Statistics
    return hss

  # Scoring option as an out put for each fold of cross validation
  scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'f1': 'f1', 'recall': 'recall', 'roc_auc': 'roc_auc', 'jaccard': 'jaccard'
  ,'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp),'fn': make_scorer(fn), 'tpr': make_scorer(tpr), 'tnr': make_scorer(tnr)
  , 'fpr': make_scorer(fpr), 'fnr': make_scorer(fnr) , 'err': make_scorer(err), 'bacc': make_scorer(bacc), 'tss': make_scorer(tss), 'hss': make_scorer(hss)}

  # Getting scores for all 
  scores = model_selection.cross_validate(model, X, y, scoring=scoring, cv=CV, return_train_score=True)


  # Creating Table for cross validation Random Forest
  if CV == None:
    print('This is performance of the '+ name +' model \n')
  else:
    print('This is performance of each cross-validation fold for '+ name +' \n')
  
  cv_header = []
  cv_header.append('Fold Num.')
  cv_header.append('accur')
  cv_header.append('prec')
  cv_header.append('f1')
  cv_header.append('recall')
  cv_header.append('roc_auc')
  cv_header.append('jaccard')
  cv_header.append('tp')
  cv_header.append('fp')
  cv_header.append('tn')
  cv_header.append('fn')
  cv_header.append('tpr')
  cv_header.append('tnr')
  cv_header.append('fpr')
  cv_header.append('fnr')
  cv_header.append('err')
  cv_header.append('bacc')
  cv_header.append('tss')
  cv_header.append('hss')
  cv_table = []

  if CV == None:
    limit = 0
  else:
    limit = CV.n_splits

  for i in range(0, limit, 1):
    temp = []
    temp.append('fold' + str(i + 1))
    temp.append(round(scores['test_accuracy'][i], 2))
    temp.append(round(scores['test_precision'][i], 2))
    temp.append(round(scores['test_f1'][i], 2))
    temp.append(round(scores['test_recall'][i], 2))
    temp.append(round(scores['test_roc_auc'][i], 2))
    temp.append(round(scores['test_jaccard'][i], 2))
    temp.append(round(scores['test_tp'][i], 2))
    temp.append(round(scores['test_fp'][i], 2))
    temp.append(round(scores['test_tn'][i], 2))
    temp.append(round(scores['test_fn'][i], 2))
    temp.append(round(scores['test_tpr'][i], 2))
    temp.append(round(scores['test_tnr'][i], 2))
    temp.append(round(scores['test_fpr'][i], 2))
    temp.append(round(scores['test_fnr'][i], 2))
    temp.append(round(scores['test_err'][i], 2))
    temp.append(round(scores['test_bacc'][i], 2))
    temp.append(round(scores['test_tss'][i], 2))
    temp.append(round(scores['test_hss'][i], 2))

    cv_table.append(temp)
  
  if CV == None:
    last_row = 'Test data'
  else:
    last_row = 'mean'
  temp = []
  temp.append(last_row)
  temp.append(round(scores['test_accuracy'].mean(), 2))
  temp.append(round(scores['test_precision'].mean(), 2))
  temp.append(round(scores['test_f1'].mean(), 2))
  temp.append(round(scores['test_recall'].mean(), 2))
  temp.append(round(scores['test_roc_auc'].mean(), 2))
  temp.append(round(scores['test_jaccard'].mean(), 2))
  temp.append(round(scores['test_tp'].mean(), 2))
  temp.append(round(scores['test_fp'].mean(), 2))
  temp.append(round(scores['test_tn'].mean(), 2))
  temp.append(round(scores['test_fn'].mean(), 2))
  temp.append(round(scores['test_tpr'].mean(), 2))
  temp.append(round(scores['test_tnr'].mean(), 2))
  temp.append(round(scores['test_fpr'].mean(), 2))
  temp.append(round(scores['test_fnr'].mean(), 2))
  temp.append(round(scores['test_err'].mean(), 2))
  temp.append(round(scores['test_bacc'].mean(), 2))
  temp.append(round(scores['test_tss'].mean(), 2))
  temp.append(round(scores['test_hss'].mean(), 2))
  cv_table.append(temp)

  print(tabulate(cv_table, cv_header, tablefmt="github"))
  print('\n')  






# Read input
df = pd.read_csv('diabetes.csv')

# Illustrating the five first rows
df.head()
# Gives information about the dataframe such as type of the variables, column names, null value counts, memory usage
df.info(verbose = True)
# Basic stats about the data
# .T at the end is for Transposing the matrix
# By default categorical values are neglected unless include = "all" is added
df.describe().T


### Cleaning Data 

# Since 0 value for Glucose, BloodPressure, SkinThickness, Insulin, and BMI	is not acceptable it shows that the values are missing. In order to count them we replace them with np.Nan
df_cleaned = df.copy(deep = True)
df_cleaned[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_cleaned[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

## showing the count of Nans
print(df_cleaned.isnull().sum())

# Replacing Nans with mean of each column
df_cleaned['Glucose'].fillna(df_cleaned['Glucose'].mean(), inplace = True)
df_cleaned['BloodPressure'].fillna(df_cleaned['BloodPressure'].mean(), inplace = True)
df_cleaned['SkinThickness'].fillna(df_cleaned['SkinThickness'].median(), inplace = True)
df_cleaned['Insulin'].fillna(df_cleaned['Insulin'].median(), inplace = True)
df_cleaned['BMI'].fillna(df_cleaned['BMI'].median(), inplace = True)

# Parameter Histogram before cleaning
p = df.hist(figsize = (20,20))

# Parameter Histogram after cleaning
p = df_cleaned.hist(figsize = (20,20))

# Pairplot of the clean data
p=sns.pairplot(df_cleaned, hue = 'Outcome')

# Plotting heatmap of correlation coefficient
plt.figure(figsize=(12,10))  
p=sns.heatmap(df_cleaned.corr(), annot=True,cmap ='crest') 

# Normalizing the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(df_cleaned.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = df_cleaned.values[:, 8]


## Modelling

# Creating train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(max_depth=3, random_state=0)
model1.fit(X_train, y_train)

# SVM Model
from sklearn import svm
model2 = svm.SVC()
model2.fit(X_train, y_train)

# Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train, y_train)

# Cross Validation
kfold = model_selection.KFold(n_splits=10, random_state = 0)

model_scoring(model1, 'Random Forest',X_train, y_train, kfold)
model_scoring(model2, 'SVM',X_train, y_train, kfold)
model_scoring(model3, 'Naive Bayes',X_train, y_train, kfold)

model_scoring(model1, 'Random Forest',X_test, y_test, None)
model_scoring(model2, 'SVM',X_test, y_test, None)
model_scoring(model3, 'Naive Bayes',X_test, y_test, None)



## BALANCING DATA

print('**** \n**** Balancing the Data \n**** \n')

temp = pd.DataFrame(y_train)
fig = temp.hist(figsize=(8,8))

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12, ratio = 1.0)
X_train_bal, y_train_bal = sm.fit_sample(X_train, y_train)


temp = pd.DataFrame(y_train_bal)
fig = temp.hist(figsize=(8,8))


# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(max_depth=3, random_state=0)
model4.fit(X_train_bal, y_train_bal)


# SVM Model
from sklearn import svm
model5 = svm.SVC()
model5.fit(X_train_bal, y_train_bal)

# Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
model6 = GaussianNB()
model6.fit(X_train_bal, y_train_bal)

model_scoring(model4, 'Random Forest',X_train_bal, y_train_bal, kfold)
model_scoring(model5, 'SVM',X_train_bal, y_train_bal, kfold)
model_scoring(model6, 'Naive Bayes',X_train_bal, y_train_bal, kfold)

model_scoring(model4, 'Random Forest',X_test, y_test, None)
model_scoring(model5, 'SVM',X_test, y_test, None)
model_scoring(model6, 'Naive Bayes',X_test, y_test, None)
