#model implementation & evaluation#
###################################

#SVM#
  #SVM_model parameter set#
C = 1.0 #제한값 
kernel = 'rbf' #비선형 장치#linear, poly, rbf, sigmoid
gamma = 0.5 #kernel 'rbf, 'poly', 'sigmoid'의 계수

  #SVM_model implementation#
clf = SVC(C = C,
          kernel = kernel,
          gamma = gamma)

clf.fit(X_train, y_train) 
test_pred = clf.predict(X_test)

  #SVM_model evaluation#
test_cm = confusion_matrix(y_test, test_pred)

test_cm_df = pd.DataFrame(test_cm,
                          index = ['M','L','H'], 
                          columns = ['M','L','H'])

plt.close()
plt.clf()
plt.figure()

plt.figure(figsize=(5.5,4))
sns.heatmap(test_cm_df, annot=True)
plt.title('SVM_test_set \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, 
                                                                  test_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() #accuracy : 0.737


#Random Forest#
  #Random Forest_model parameter set#
n_estimators = 100 #나무의 개수
max_features = 6 #특성의 수
min_samples_leaf = 6 #특성의 수
bootstrap = False #부트 스트래핑 허용 여부
  
  #Random Forest_model implementation#
clf = RandomForestClassifier(n_estimators = n_estimators,
                             max_features = max_features,
                             min_samples_leaf = min_samples_leaf,
                             bootstrap = bootstrap)

clf.fit(X_train, y_train) 
test_pred = clf.predict(X_test)

  #Random Forest_model evaluation#
test_cm = confusion_matrix(y_test, test_pred)

test_cm_df = pd.DataFrame(test_cm,
                          index = ['M','L','H'], 
                          columns = ['M','L','H'])

plt.close()
plt.clf()
plt.figure()

plt.figure(figsize=(5.5,4))
sns.heatmap(test_cm_df, annot=True)
plt.title('Random Forest_test_set \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, 
                                                                            test_pred)))
plt.ylabel('True label') 
plt.xlabel('Predicted label')
plt.show() #accuracy : 0.736


#XGBoost#
  #Random Forest_model parameter set#
objective = "multi:softprob" #목적
gamma_xg = 0.1 #최소 손실 축소 # 나무 생성 멈추는 기준
minmin_child_weight = 1 #나무 잎사귀 노드에 주는 최소 가중값(표본들),
max_depth = 12 #상호작용 나무 개수
learlearning_rate = 0.01 #학습률
n_estn_estimators = 500 #나무의 개수

  #_model implementation#
clf = xgb.XGBClassifier(objective = objective,
                        gamma = gamma_xg, 
                        minmin_child_weight = minmin_child_weight,
                        max_depth = max_depth, 
                        learlearning_rate = learlearning_rate, 
                        n_estn_estimators = n_estn_estimators)

clf.fit(X_train, y_train) 
test_pred = clf.predict(X_test)

  #XGBoost_model evaluation#
test_cm = confusion_matrix(y_test, test_pred)

test_cm_df = pd.DataFrame(test_cm,
                          index = ['M','L','H'], 
                          columns = ['M','L','H'])

plt.close()
plt.clf()
plt.figure()

plt.figure(figsize=(5.5,4))
sns.heatmap(validation_cm_df, annot=True)
plt.title('XGBoost_test_set \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, 
                                                                      test_pred)))
plt.ylabel('True label') 
plt.xlabel('Predicted label')
plt.show() #accuracy : 0.734


#ensemble methods#
  #ensemble methods implementation#
clf_1 = SVC(C = C,
          kernel = kernel,
          gamma = gamma)
clf_2 = RandomForestClassifier(n_estimators = n_estimators,
                             max_features = max_features,
                             min_samples_leaf = min_samples_leaf,
                             bootstrap = bootstrap)
clf_3 = xgb.XGBClassifier(objective = objective,
                        gamma = gamma_xg, 
                        minmin_child_weight = minmin_child_weight,
                        max_depth = max_depth, 
                        learlearning_rate = learlearning_rate, 
                        n_estn_estimators = n_estn_estimators)

eclf1 = VotingClassifier(estimators=[('svm', clf_1), ('rf', clf_2), ('xgb', clf_3)], 
                         voting='hard')
                         
eclf1.fit(X_train, y_train) 
test_pred = eclf1.predict(X_test)

  #ensemble methods evaluation#
test_cm = confusion_matrix(y_test, test_pred)

test_cm_df = pd.DataFrame(test_cm,
                          index = ['M','L','H'], 
                          columns = ['M','L','H'])

plt.close()
plt.clf1()
plt.figure()

plt.figure(figsize=(5.5,4))
sns.heatmap(test_cm_df, annot=True)
plt.title('ensemble methods_test_set \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, 
                                                                               test_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() #accuracy : 0.737
