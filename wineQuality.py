#!pip install imbalanced-learn scikit-learn


#!pip install scikit-learn==1.0.2
#!pip install imbalanced-learn==0.10.1



import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression , LogisticRegression
import matplotlib as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error, mean_absolute_percentage_error, r2_score ,  classification_report
from sklearn.ensemble  import RandomForestRegressor , RandomForestClassifier
# ! pip install py3Dmol
# import py3Dmol
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import graphviz
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE


from sklearn.neighbors import KNeighborsClassifier
#!pip install auto-sklearn
#!pip install tpot



# p = py3Dmol.view(query = 'mmtf:1ycr')
# p.setStyle({'cartoon' : {'color' : 'spectrum'} })

# rw = pd.read_csv("https://raw.githubusercontent.com/miconunogluali/Wine_Quality/main/winequality-red.csv")
rw = pd.read_csv("https://raw.githubusercontent.com/kelebekkadircan/wineQuality_prediction/main/winequality.csv?token=GHSAT0AAAAAACPQQCRIBPSDXVK75F2GATNWZSAYFQA")
rw = rw.drop("type" , axis = 1)
rw
rw.head(15)
# This code reads a CSV file from the URL specified with the "read_csv()" function using the pandas library.

preprocessing

rw.columns=["sabitasit","degiskenasit","sitrikasit","atikseker","klorur","serbestsulfur","totalsulfur","yogunluk","ph","sulfat","alkol","kalite"]
rw["kalite"] =  rw["kalite"].astype(float)
#  pd.to_numeric(rw["kalite"])
# rw["kalite"].astype(float)

# fill the missing values
for col, value in rw.items():
    if col != 'tip':
        rw[col] = rw[col].fillna(rw[col].mean())

print(rw)

y= rw[["kalite"]]
x= rw.drop(columns=["kalite"],axis=1)

# class ımbalancement
oversample = SMOTE(k_neighbors=4)

# transform the dataset
x, y = oversample.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

rw

# number of values for each quality
sns.catplot(x='kalite', data = rw, kind = 'count')

modelTraining

def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    # train the model
    model.fit(x_train, y_train)
    print("Accuracy:", model.score(x_test, y_test) * 100)

    y_lr_train_pred = model.predict(x_train)
    y_lr_test_pred = model.predict(x_test)


    lr_train_mse = mean_squared_error(y_train , y_lr_train_pred)
    lr_train_mae = mean_absolute_error(y_train, y_lr_train_pred )
    lr_train_mape = mean_absolute_percentage_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train , y_lr_train_pred)

    lr_test_mse = mean_squared_error(y_test , y_lr_test_pred)
    lr_test_mae = mean_absolute_error(y_test , y_lr_test_pred)
    lr_test_mape =  mean_absolute_percentage_error(y_test , y_lr_test_pred)
    lr_test_r2 = r2_score(y_test , y_lr_test_pred)

    lr_results = pd.DataFrame([  lr_train_mse,  lr_train_r2 , lr_test_mse ,  lr_test_r2]).transpose()
    lr_results.columns = [ 'Training MSE' , 'Training R2' , 'Test MSE' , 'Test R2']
    print(lr_results)

#**dist plot show**

# create dist plot
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in rw.items():
    if col != 'tip':
        sns.distplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

logTransformation

rw["serbestsulfur"] = np.log( 1 + rw['serbestsulfur'])

sns.distplot(rw['serbestsulfur'])

denemeregression

corr = rw.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')

bins = [0, 5.5, 7.5, 10] # this means 3-5 are low, 6-7 are mid, 8-9 are high
labels = [0, 1, 2]
rw['kalite'] = pd.cut(rw['kalite'], bins=bins, labels=labels)

rw.head(15)

#K Nearest Neighbors Classifier Trying

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import pandas as pd

# KNN modeli oluşturma
knn = KNeighborsClassifier(n_neighbors=3)  # Komşu sayısını 3 olarak ayarladık

# Modeli eğitme
knn.fit(x_train, y_train)

# Test verileri üzerinde tahmin yapma
y_pred = knn.predict(x_test)

# Sınıflandırma doğruluğunu hesaplama ve yazdırma
accuracy = accuracy_score(y_test, y_pred)
print("Test verileri doğruluğu:", accuracy)

# Sınıflandırma raporunu yazdırma
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Belirli bir örneğe göre tahmin yapma
example = [[7.4,0.70,0.00,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]]
prediction = knn.predict(example)
print("\nÖrnek için Tahmin:", prediction)

# Eğitim verileri üzerinde tahmin yapma
y_train_pred = knn.predict(x_train)

# Eğitim ve test verileri üzerinde MSE hesaplama
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_pred)

# Eğitim ve test verileri üzerinde R2 hesaplama
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)

# KNN sınıflandırıcının sonuçlarını DataFrame'e kaydetme
knn_results = pd.DataFrame([['KNN Classifier', accuracy, train_mse, train_r2, test_mse, test_r2]], columns=['Method', 'Test Score', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2'])
knn_results


#ANN Classifier Trying

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Yapay Sinir Ağı modeli oluşturma
model = Sequential()

# Modelin katmanlarını ekleme
model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))  # Giriş katmanı
model.add(Dense(units=64, activation='relu'))  # Gizli katman
model.add(Dense(units=1, activation='linear'))  # Çıkış katmanı

# Modeli derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Eğitim ve test verileri üzerinde tahmin yapma
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Eğitim ve test verileri üzerinde MSE ve R2 hesaplama
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Modelin sonuçlarını DataFrame'e kaydetme
ann_results = pd.DataFrame([['Artificial Neural Network', None, train_mse, train_r2, test_mse, test_r2]], columns=['Method', 'Test Score', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2'])
ann_results


#**LinearRegression**

lm = LinearRegression()
newmodel = lm.fit(x_train,y_train)

# print(newmodel.score(x_test,y_test))
# print(newmodel.score(x_train,y_train))

lrTestScore = newmodel.score(x_test,y_test)
lrTrainScore = newmodel.score(x_train,y_train)

y_lr_train_pred = newmodel.predict(x_train)
y_lr_test_pred = newmodel.predict(x_test)

# print(y_lr_train_pred)
# print(y_lr_test_pred)

lr_train_mse = mean_squared_error(y_train , y_lr_train_pred)
lr_train_mae = mean_absolute_error(y_train, y_lr_train_pred )
lr_train_mape = mean_absolute_percentage_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train , y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test , y_lr_test_pred)
lr_test_mae = mean_absolute_error(y_test , y_lr_test_pred)
lr_test_mape =  mean_absolute_percentage_error(y_test , y_lr_test_pred)
lr_test_r2 = r2_score(y_test , y_lr_test_pred)

lr_results = pd.DataFrame(['Linear Regression' ,lrTestScore,  lr_train_mse,  lr_train_r2 , lr_test_mse ,  lr_test_r2]).transpose()
lr_results.columns = ['Method' ,"Test Score" , 'Training MSE' , 'Training R2' , 'Test MSE' , 'Test R2']

print(lr_results)

prediction = newmodel.predict([[7.4,0.70,0.00,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]])[0][0]

print("Tahmin:", prediction)

#**LogisticRegression**

lr = LogisticRegression(max_iter=400)
logModel = lr.fit(x_train , y_train)
print(logModel.score(x_test,y_test))
print(logModel.score(x_train,y_train))
logTestScore  = logModel.score(x_test,y_test)
logTrainScore = logModel.score(x_train,y_train)

y_log_train_pred = logModel.predict(x_train)
y_log_test_pred = logModel.predict(x_test)

log_train_mse = mean_squared_error(y_train , y_log_train_pred)
log_train_mae = mean_absolute_error(y_train, y_log_train_pred )
log_train_mape = mean_absolute_percentage_error(y_train, y_log_train_pred)
log_train_r2 = r2_score(y_train , y_log_train_pred)


log_test_mse = mean_squared_error(y_test , y_log_test_pred)
log_test_mae = mean_absolute_error(y_test , y_log_test_pred)
log_test_mape =  mean_absolute_percentage_error(y_test , y_log_test_pred)
log_test_r2 = r2_score(y_test , y_log_test_pred)

log_results = pd.DataFrame(['Logistic Regression' ,logTestScore, log_train_mse,  log_train_r2 , log_test_mse ,  log_test_r2]).transpose()
log_results.columns =['Method' ,"Test Score" , 'Training MSE' , 'Training R2' , 'Test MSE' , 'Test R2']

print(log_results)


prediction = logModel.predict([[7.4,0.70,0.00,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]])

print("Tahmin:", prediction)

randomForestRegressor


rf = RandomForestRegressor(max_depth = 5, n_estimators=200)
forestModel = rf.fit(x_train , y_train)

print(forestModel.score(x_test,y_test))
# print(forestModel.score(x_train,y_train))

rfTestScore = forestModel.score(x_test,y_test)
rfTrainScore = forestModel.score(x_train,y_train)

y_rf_train_pred = forestModel.predict(x_train)
y_rf_test_pred = forestModel.predict(x_test)

rf_train_mse = mean_squared_error(y_train , y_rf_train_pred)
rf_train_mae = mean_absolute_error(y_train, y_rf_train_pred )
rf_train_mape = mean_absolute_percentage_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train , y_rf_train_pred)


rf_test_mse = mean_squared_error(y_test , y_rf_test_pred)
rf_test_mae = mean_absolute_error(y_test , y_rf_test_pred)
rf_test_mape =  mean_absolute_percentage_error(y_test , y_rf_test_pred)
rf_test_r2 = r2_score(y_test , y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest Regressor' ,rfTestScore, rf_train_mse, rf_train_r2 , rf_test_mse , rf_test_r2]).transpose()
rf_results.columns =['Method' ,"Test Score" , 'Training MSE' ,'Training R2' , 'Test MSE' , 'Test R2']

print(rf_results)

prediction = forestModel.predict([[7.4,0.70,0.00,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]])

print("Tahmin:", prediction)

decisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=10)
dtModel = dtc.fit(x_train,y_train)

dtcTestScore = dtModel.score(x_test,y_test)
dtcTrainScore = dtModel.score(x_train,y_train)

y_dtc_train_pred = dtModel.predict(x_train)
y_dtc_test_pred = dtModel.predict(x_test)

dtc_train_mse = mean_squared_error(y_train , y_dtc_train_pred)
dtc_train_mae = mean_absolute_error(y_train, y_dtc_train_pred )
dtc_train_mape = mean_absolute_percentage_error(y_train, y_dtc_train_pred)
dtc_train_r2 = r2_score(y_train , y_dtc_train_pred)

dtc_test_mse = mean_squared_error(y_test , y_dtc_test_pred)
dtc_test_mae = mean_absolute_error(y_test , y_dtc_test_pred)
dtc_test_mape =  mean_absolute_percentage_error(y_test , y_dtc_test_pred)
dtc_test_r2 = r2_score(y_test , y_dtc_test_pred)

dtc_results = pd.DataFrame(['Decision Tree Classifier' ,dtcTestScore, dtc_train_mse,  dtc_train_r2 , dtc_test_mse ,  dtc_test_r2]).transpose()
dtc_results.columns =['Method' ,"Test Score" , 'Training MSE' , 'Training R2' , 'Test MSE' , 'Test R2']

print(dtc_results)

prediction = dtModel.predict([[7.4,0.70,0.00,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]])
print("Tahmin:", prediction)

report = classification_report(y_test,y_dtc_test_pred)
print(report)



#**randomforestclassfier**

rfc=RandomForestClassifier(n_estimators = 400 ,max_depth=10)
rfcModel = rfc.fit(x_train, y_train)

rfcTestScore = rfcModel.score(x_test,y_test)
rfcTrainScore = rfcModel.score(x_train,y_train)
print(rfcModel.score(x_test,y_test))
# print(rfcModel.score(x_train,y_train))

y_rfc_train_pred = rfcModel.predict(x_train)
y_rfc_test_pred = rfcModel.predict(x_test)

rfc_train_mse = mean_squared_error(y_train , y_rfc_train_pred)
rfc_train_mae = mean_absolute_error(y_train, y_rfc_train_pred )
rfc_train_mape = mean_absolute_percentage_error(y_train, y_rfc_train_pred)
rfc_train_r2 = r2_score(y_train , y_rfc_train_pred)

rfc_test_mse = mean_squared_error(y_test , y_rfc_test_pred)
rfc_test_mae = mean_absolute_error(y_test , y_rfc_test_pred)
rfc_test_mape =  mean_absolute_percentage_error(y_test , y_rfc_test_pred)
rfc_test_r2 = r2_score(y_test , y_rfc_test_pred)

rfc_results = pd.DataFrame(['Random Forest Classifier' ,rfcTestScore, rfc_train_mse,  rfc_train_r2 , rfc_test_mse ,  rfc_test_r2]).transpose()
rfc_results.columns =['Method' ,"Test Score" , 'Training MSE' ,'Training R2' , 'Test MSE' ,'Test R2']

print(rfc_results)

prediction = dtModel.predict([[7.4,0.70,0.00,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]])
print("Tahmin:", prediction)

report = classification_report(y_test,y_rfc_test_pred)
print(report)



#GENEL **TABLO**

df_models  = pd.concat([lr_results , log_results,dtc_results , rf_results, rfc_results, knn_results,ann_results], axis = 0)
df_models

#**DECISION TREE Görselleştirme**

dot = export_graphviz(dtModel , feature_names=x_test.columns , filled=True)
gorsel  = graphviz.Source(dot)
gorsel

# Visualization for Linear Regression
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_lr_train_pred, color='blue', label='Train')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', lw=2)  # Diagonal line
plt.xlabel('Actual Values')
plt.ylabel('Estimated Values')
plt.title('Linear Regression: Real vs Prediction (Train)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_lr_test_pred, color='red', label='Test')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)  # Diagonal line
plt.xlabel('Actual Values')
plt.ylabel('Estimated Value')
plt.title('Linear Regression: Real vs Prediction (Test)')
plt.legend()

plt.tight_layout()
plt.show()

# Visualization for Random Forest
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_rf_train_pred, color='blue', label='Train')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', lw=2)  # Diagonal line
plt.xlabel('Actual Values')
plt.ylabel('Tahmin Estimated Value')
plt.title('Random Forest: Real vs Prediction (Train)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_rf_test_pred, color='red', label='Test')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)  # Diagonal line
plt.xlabel('Actual Value')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Random Forest: Gerçek vs Tahmin (Test)')
plt.legend()

plt.tight_layout()
plt.show()

model = LinearRegression()
classify(model,x,y)

model = LogisticRegression()
classify(model,x,y)

model = DecisionTreeClassifier()
classify(model,x,y)

model = RandomForestClassifier()
classify(model,x,y)

model = RandomForestRegressor()
classify(model,x,y)

