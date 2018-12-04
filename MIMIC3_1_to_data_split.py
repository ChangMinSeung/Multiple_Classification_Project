
#libraries_installed#
#####################
import pandas as pd #data processing
from pandas import DataFrame #pandas dataframe
import numpy as np #numpy
import seaborn as sns #data visualization library  
import matplotlib.pyplot as plt #data visualization library
import statsmodels.api as sm #QQplot
import pylab #QQplot
import scipy as sp #anderson-darling normality test
from sklearn.preprocessing import minmax_scale, scale #scaling
from sklearn.decomposition import PCA #pca
from sklearn.model_selection import train_test_split #data split
from sklearn.model_selection import KFold #k-fold cross validation
from sklearn.svm import SVC #svm model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support #metrics
from sklearn.ensemble import RandomForestClassifier #Random Forest model
import xgboost as xgb #XGBoost_model 
from sklearn.ensemble import VotingClassifier #ensemble methods #aggregation(취합) #Majority Voting(다수결)
from sklearn.model_selection import GridSearchCV #GridSearch cross validation


#data_import#
#############
MIMIC = pd.read_csv('E:/2018_FALL/데이터분석언어/기말_프로젝트/MIMIC3_DATA/MIMIC_FINAL_RE2.csv')
MIMIC.head


#EDA#
#####
MIMIC.shape #1만 1,994명의 환자와 2개의 이환정보변수, 13개의 독립변수, 1개의 반응변수

col = MIMIC.columns
print(col) 

MIMIC.info() #정보요약
MIMIC.iloc[:,2:15].describe() #독립변수 기초 통계
MIMIC.count() #결측치 제외 개수 확인
  #M_8,849명(74%) / L_1,937명(16%) / H_1,208명(10%) -> Resampling_Over-Sampling_고려
MIMIC.target.value_counts() 
def percentage(part, whole): 
    return 100 * float(part)/float(whole)
round(percentage(8849, 11994))
round(percentage(1937, 11994))
round(percentage(1208, 11994))

#correlation
  #sodium-chloride(0.70)
plt.close()
plt.clf()
plt.figure()
sns.heatmap(data=MIMIC[['chloride', 'elixhauser_vanwalraven_avg', 'elixhauser_vanwalraven_max',
                        'elixhauser_vanwalraven_min', 'glucose', 'hb', 'platelet', 'potassium',
                        'sbp', 'sodium', 'temperature']].corr(), annot=True, fmt= '.2f',
                        linewidths= 2, cmap= 'Blues', square=True) 
plt.xticks(rotation=14, ha='right')
plt.title('Correlation')
plt.show()
  #target기준
MIMIC_tg_0 = MIMIC[MIMIC.target == "M"]  
  #sodium-chloride(0.68)
plt.close()
plt.clf()
plt.figure()
sns.heatmap(data=MIMIC_tg_0[['chloride', 'elixhauser_vanwalraven_avg', 'elixhauser_vanwalraven_max',
                             'elixhauser_vanwalraven_max', 'elixhauser_vanwalraven_min', 'glucose',
                             'hb', 'platelet', 'potassium', 'sbp', 'sodium', 'temperature']].corr(), 
                             annot=True, fmt= '.2f', linewidths= 2, cmap= 'Blues', square=True) 
plt.xticks(rotation=14, ha='right')
plt.title('Target_M(Creatinine Level Maintain) Correlation')
plt.show()

MIMIC_tg_1 = MIMIC[MIMIC.target == "L"] 
  #sodium-chloride(0.80)
plt.close()
plt.clf()
plt.figure()
sns.heatmap(data=MIMIC_tg_1[['chloride', 'elixhauser_vanwalraven_avg', 'elixhauser_vanwalraven_max',
                             'elixhauser_vanwalraven_max', 'elixhauser_vanwalraven_min', 'glucose',
                             'hb', 'platelet', 'potassium', 'sbp', 'sodium', 'temperature']].corr(), 
                             annot=True, fmt= '.2f', linewidths= 2, cmap= 'Blues', square=True) 
plt.xticks(rotation=14, ha='right')
plt.title('Target_L(Creatinine Level Downward) Correlation')
plt.show()

MIMIC_tg_2 = MIMIC[MIMIC.target == "H"] 
  #sodium-chloride(0.6)
plt.close()
plt.clf()
plt.figure()
sns.heatmap(data=MIMIC_tg_2[['chloride', 'elixhauser_vanwalraven_avg', 'elixhauser_vanwalraven_max',
                             'elixhauser_vanwalraven_max', 'elixhauser_vanwalraven_min', 'glucose',
                             'hb', 'platelet', 'potassium', 'sbp', 'sodium', 'temperature']].corr(), 
                             annot=True, fmt= '.2f', linewidths= 2, cmap= 'Blues', square=True) 
plt.xticks(rotation=14, ha='right')
plt.title('Target_H(Creatinine Level Lift) Correlation')
plt.show()

#normality_test
  #age #300살 -> 100세 이상을 Raw Data에서 임의적으로 300으로 기록
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['age']]), rug=True)
plt.title('Feature Age Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['age']]))
plt.title('BoxPlot')
plt.grid()

plt.show()

  #chloride
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['chloride']]), rug=True)
plt.title('Feature chloride Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['chloride']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['chloride'], loc = 4, scale = 3, line='s')
plt.title('chloride QQplot')
pylab.show()
#Anderson-Darling Normality Test -> 통계량이 Critical_values보다 더 크기 떄문에 귀무가설 기각
sp.stats.anderson(MIMIC['chloride'], dist='norm')

  #elixhauser_vanwalraven_avg
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['elixhauser_vanwalraven_avg']]), rug=True)
plt.title('Feature elixhauser_vanwalraven_avg Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['elixhauser_vanwalraven_avg']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['elixhauser_vanwalraven_avg'], loc = 4, scale = 3, line='s')
plt.title('elixhauser_vanwalraven_avg QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['elixhauser_vanwalraven_avg'], dist='norm')

  #elixhauser_vanwalraven_max
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['elixhauser_vanwalraven_max']]), rug=True)
plt.title('Feature elixhauser_vanwalraven_max Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['elixhauser_vanwalraven_max']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['elixhauser_vanwalraven_max'], loc = 4, scale = 3, line='s')
plt.title('elixhauser_vanwalraven_max QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['elixhauser_vanwalraven_max'], dist='norm')

  #elixhauser_vanwalraven_min
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['elixhauser_vanwalraven_min']]), rug=True)
plt.title('Feature elixhauser_vanwalraven_min Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['elixhauser_vanwalraven_min']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['elixhauser_vanwalraven_min'], loc = 4, scale = 3, line='s')
plt.title('elixhauser_vanwalraven_min QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['elixhauser_vanwalraven_min'], dist='norm')

  #glucose #Positive skewed
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['glucose']]), rug=True)
plt.title('Feature glucose Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['glucose']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['glucose'], loc = 4, scale = 3, line='s')
plt.title('glucose QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['glucose'], dist='norm')

  #hb
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['hb']]), rug=True)
plt.title('Feature hb Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['hb']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['hb'], loc = 4, scale = 3, line='s')
plt.title('hb QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['hb'], dist='norm')

  #platelet #Positive skewed 
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['platelet']]), rug=True)
plt.title('Feature platelet Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['platelet']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['platelet'], loc = 4, scale = 3, line='s')
plt.title('platelet QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['platelet'], dist='norm')

  #potassium 
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['potassium']]), rug=True)
plt.title('Feature potassium Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['potassium']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['potassium'], loc = 4, scale = 3, line='s')
plt.title('potassium QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['potassium'], dist='norm')

  #rr
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['rr']]), rug=True)
plt.title('Feature rr Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['rr']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['rr'], loc = 4, scale = 3, line='s')
plt.title('rr QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['rr'], dist='norm')

  #sbp
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['sbp']]), rug=True)
plt.title('Feature sbp Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['sbp']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['sbp'], loc = 4, scale = 3, line='s')
plt.title('sbp QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['sbp'], dist='norm')

  #sodium
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['sodium']]), rug=True)
plt.title('Feature sodium Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['sodium']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['sodium'], loc = 4, scale = 3, line='s')
plt.title('sodium QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['sodium'], dist='norm')

  #temperature #Negative skewed
plt.close()
plt.clf()
plt.figure()

plt.subplot(121)
sns.distplot(round(MIMIC[['temperature']]), rug=True)
plt.title('Feature temperature Histogram')
plt.grid()

plt.subplot(122)
sns.boxplot(data = round(MIMIC[['temperature']]))
plt.title('BoxPlot')
plt.grid()

plt.show()
#QQplot
plt.close()
plt.clf()
plt.figure()

sm.qqplot(MIMIC['temperature'], loc = 4, scale = 3, line='s')
plt.title('temperature QQplot')
pylab.show()
#Anderson-Darling Normality Test
sp.stats.anderson(MIMIC['temperature'], dist='norm')


#min-max scaling# 
#################
MIMIC_x = MIMIC[['chloride', 'elixhauser_vanwalraven_avg', 'elixhauser_vanwalraven_max',
                 'elixhauser_vanwalraven_min', 'glucose', 'hb', 'platelet', 'potassium',
                 'rr', 'sbp', 'sodium', 'temperature']]
MIMIC_x.info()
MIMIC_x = MIMIC_x.astype(np.float64) #정수형(int64)을 부동소수형(float64)으로 변환
MIMIC_x.info()

Scaled_MIMIC_x = minmax_scale(MIMIC_x)

Scaled_MIMIC_x_df = pd.DataFrame(Scaled_MIMIC_x, 
                                 columns=['chloride', 'elixhauser_vanwalraven_avg', 
                                          'elixhauser_vanwalraven_max',
                                          'elixhauser_vanwalraven_min', 'glucose', 'hb', 
                                          'platelet', 'potassium', 'rr', 'sbp', 
                                          'sodium', 'temperature'])
Scaled_MIMIC_x_df.describe()


#PCA#
#####
X = Scaled_MIMIC_x

pca = PCA(n_components=12) #12개의 인자수(피처의 수)

pca.fit(X)

var = pca.explained_variance_ratio_ #각 pca가 설명하는 분산의 양

var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100) #누적 분산에 대한 설명

print(var1) #32.88/46.77/58.17/68.13/77.94/85.78/90.44/93.79/96.94/98.69/99.93/100.01(분산의 설명력)
            #5개 주성분에서 elbow point 확인 -> 급격한 기울기 변환을 보이는 부분 x 
            #상관관계가 낮은 변수들로 정규분포를 띄지 않는 데이터 셋이기 때문에 차원 축소 X
plt.close()
plt.clf()
plt.figure()

plt.plot(var1, marker='o', label='n_components=12')
plt.legend(loc='upper left')
plt.title('Scree plot')
plt.show()


#data split#
############
series_icuId = pd.DataFrame(MIMIC[['icuId']])
series_age = pd.DataFrame(MIMIC[['age']])
series_target = pd.DataFrame(MIMIC[['target']])

tmp1 = pd.concat([series_age, Scaled_MIMIC_x_df], axis=1)
tmp2 = pd.concat([series_icuId, tmp1], axis=1)
MIMIC_2 = pd.concat([tmp2, series_target], axis=1)

print(MIMIC_2) #scaling된 dataset
MIMIC_2.info()

df = Scaled_MIMIC_x_df  
y = series_target  
tmp3 = pd.concat([df, y], axis=1)
tmp4 = pd.concat([tmp1, y], axis=1)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = 0)
print(X_train.shape, y_train.shape) #train set (9595, 12) (9595, 1)
print(X_test.shape, y_test.shape) #test set (2399, 12) (2399, 1)

X_train_2, X_train_validation, y_train_2, y_train_validation = train_test_split(X_train, y_train, 
                                                                                test_size = 0.2,
                                                                                random_state = 0)
print(X_train_2.shape, y_train_2.shape) #train set (7676, 12) (7676, 1)
print(X_train_validation.shape, y_train_validation.shape) #test set (1919, 12) (1919, 1)

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(tmp1, y, test_size = 0.2, random_state = 0)
print(X_train_3.shape, y_train_3.shape) #train set (9595, 12) (9595, 1)
print(X_test_3.shape, y_test_3.shape) #test set (2399, 12) (2399, 1)

