import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

'''
OSS Project 2-2
Author: 12183588 컴퓨터공학과 허대현
'''

def sort_dataset(dataset_df):
	return dataset_df.sort_values(by='year') # sort_values 이용해, 연도 기준으로 오름차 정렬 정렬.

def split_dataset(dataset_df):
	dataset_df:pd.DataFrame = dataset_df
	data = data_df.drop(columns='salary', axis=1) # salary column을 제거한 데이터를 추출한다.
	label = data_df['salary']*0.001 # salary 컬럼만 추출하여 리스케일 해준다
	x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1015, shuffle=False) # train 데이터 사이즈 1718, test 데이터 사이즈 195 
	return x_train, x_test, y_train, y_test #한꺼번에 리턴한다.

def extract_numerical_cols(dataset_df):
	dataset_df:pd.DataFrame = dataset_df
	return dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]
    #필요한 column 만 고른 데이터프레임만 리턴한다. 

def train_predict_decision_tree(X_train, Y_train, X_test):
	dt_pipeline = make_pipeline( StandardScaler(), DecisionTreeRegressor()) # 파이프라인을 이용해서 짠다
	dt_pipeline.fit(X_train, Y_train)
	return dt_pipeline.predict(X_test)
	

def train_predict_random_forest(X_train, Y_train, X_test):
	rf_pipeline = make_pipeline( StandardScaler(), RandomForestRegressor())
	rf_pipeline.fit(X_train, Y_train)
	return rf_pipeline.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):
	svm_pipeline = make_pipeline( StandardScaler(), SVR())
	svm_pipeline.fit(X_train, Y_train)
	return svm_pipeline.predict(X_test)

def calculate_RMSE(labels, predictions):
	return np.sqrt(np.mean((predictions-labels)**2)) # RMSE를 구한다. 

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))