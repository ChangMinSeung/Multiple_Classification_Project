
#model implementation & evaluation#
###################################

#SVM#
  #SVM_model implementation#
model = SVC()
param_grid = {"gamma": np.logspace(-6, -1, 10)}
gs1 = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=1)

gs1.fit(X_train, y_train)
test_pred = gs1.predict(X_test)

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
plt.title('SVM_grid-search_test_set \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, test_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() #accuracy : 0.737


#XGBoost#
model = xgb.XGBClassifier()
param_grid = {"gamma": np.logspace(-6, -1, 10)}
gs1 = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=1)

gs1.fit(X_train, y_train)
test_pred = gs1.predict(X_test)

#5. grid_search_cv
test_cm = confusion_matrix(y_test, test_pred)

test_cm_df = pd.DataFrame(test_cm,
                          index = ['M','L','H'], 
                          columns = ['M','L','H'])

plt.close()
plt.clf()
plt.figure()

plt.figure(figsize=(5.5,4))
sns.heatmap(test_cm_df, annot=True)
plt.title('XGBoost_grid-search_validation_set \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, test_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() #accuracy : 0.737


#########plus age feature##########
#model implementation & evaluation#
###################################

#SVM#
  #SVM_model implementation#
model = SVC()
param_grid = {"gamma": np.logspace(-6, -1, 10)}
gs1 = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=1)

gs1.fit(X_train_3, y_train_3)
test_pred = gs1.predict(X_test_3)

  #SVM_model evaluation#
test_cm = confusion_matrix(y_test_3, test_pred)

test_cm_df = pd.DataFrame(test_cm,
                          index = ['M','L','H'], 
                          columns = ['M','L','H'])

plt.close()
plt.clf()
plt.figure()

plt.figure(figsize=(5.5,4))
sns.heatmap(test_cm_df, annot=True)
plt.title('SVM_grid-search_test_set \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, test_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() #accuracy : 0.737


#XGBoost#
model = xgb.XGBClassifier()
param_grid = {"gamma": np.logspace(-6, -1, 10)}
gs1 = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=1)

gs1.fit(X_train_3, y_train_3)
test_pred = gs1.predict(X_test_3)

#5. grid_search_cv
test_cm = confusion_matrix(y_test_3, test_pred)

test_cm_df = pd.DataFrame(test_cm,
                          index = ['M','L','H'], 
                          columns = ['M','L','H'])

plt.close()
plt.clf()
plt.figure()

plt.figure(figsize=(5.5,4))
sns.heatmap(test_cm_df, annot=True)
plt.title('XGBoost_grid-search_validation_set \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, test_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() #accuracy : 0.737
