from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.linear_model import (
	LinearRegression,
	LogisticRegression,
	Ridge,
	Lasso,
	ElasticNet,
)

from sklearn.preprocessing import (
	MinMaxScaler,
	StandardScaler,
)

from sklearn.model_selection import (
	RandomizedSearchCV,
	PredefinedSplit,
)

from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
	RandomForestClassifier,
	RandomForestRegressor,
)

from sklearn.metrics import (
	max_error,
	mean_absolute_error,
	mean_squared_error,
	r2_score,
	accuracy_score,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb

from config import *

SEED = 42
np.random.seed(SEED)

def one_hot_df(df, columns):
	for column in columns:
		dummies = pd.get_dummies(df[column], prefix=column, drop_first=False)
		df = pd.concat([df, dummies], axis=1)
	df = df.drop(columns, axis=1)
	return df

def min_max_scaler_df(df, columns):
	for column in columns:
		df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
	return df

def normalize_df(df, columns):
	for column in columns:
		df[column] = (df[column] - df[column].mean()) / df[column].std()
	return df

def get_model(model_type='linear', task_type='regression'):
	model = None
	if task_type == 'regression':
		if model_type == 'linear':
			model = LinearRegression(random_state=SEED)
		elif model_type == 'ridge':
			model = Ridge(random_state=SEED)
		elif model_type == 'lasso':
			model = Lasso(random_state=SEED)
		elif model_type == 'elastic_net':
			model = ElasticNet(random_state=SEED)
		elif model_type == 'random_forest':
			model = RandomForestRegressor(random_state=SEED)
		elif model_type == 'xgboost':
			model = xgb.XGBRegressor(objective='reg:squarederror', random_state=SEED)
		elif model_type == 'svr':
			model = SVR()
		elif model_type == 'mlp':
			model = MLPRegressor(random_state=SEED)
	elif task_type == 'classification':
		if model_type == 'random_forest':
			model = RandomForestClassifier(class_weight='balanced_subsample', random_state=SEED)
		elif model_type == 'xgboost':
			model = xgb.XGBClassifier(objective='binary:hinge', random_state=SEED)
		elif model_type == 'logistic':
			model = LogisticRegression(random_state=SEED)
	return model

def get_param_grid(model_type='linear', task_type='regression'):
	param_grid = {}
	if task_type == 'regression':
		if (model_type == 'ridge') or (model_type == 'lasso'):
			param_grid = {
				'regressor__alpha': stats.uniform.rvs(loc=0, scale=4, size=3),
				'regressor__normalize': [True, False],
			}
		if model_type == 'elastic_net':
			param_grid = {
				'regressor__alpha': stats.uniform.rvs(loc=0, scale=4, size=3),
				'regressor__l1_ratio': np.linspace(0, 1, num=10),
				'regressor__normalize': [True, False],
			}
		elif model_type == 'random_forest':
			param_grid = {
				'regressor__n_estimators': range(200, 2000),
				'regressor__max_features': ['auto', 'sqrt', 'log2'],
				'regressor__max_depth': range(3, 10),
				'regressor__min_samples_split': range(2, 10),
				'regressor__min_samples_leaf': range(1, 10),
				'regressor__bootstrap': [True, False],
			}
		elif model_type == 'xgboost':
			param_grid = {
				'regressor__n_estimators': stats.randint(200, 2000),
				'regressor__learning_rate': np.random.uniform(1e-3, 0.2, 100),
				'regressor__subsample': np.random.uniform(0.9, 1, 100),
				'regressor__max_depth': stats.randint(3, 10),
				'regressor__colsample_bytree': np.random.uniform(0.7, 1, 100),
				'regressor__min_child_weight': stats.randint(1, 5),
				'regressor__gamma': np.random.uniform(0.5, 5, 100),
			}
		elif model_type == 'svr':
			param_grid = {
				'regressor__kernel': ['linear', 'poly', 'rbf'],
				'regressor__degree': range(1, 5),
				'regressor__gamma': ['auto', 'scale'],
				'regressor__C': np.linspace(0, 10, 100),
			}
	elif task_type == 'classification':
		if model_type == 'random_forest':
			param_grid = {
				'classifier__n_estimators': range(200, 2000),
				'classifier__max_features': ['auto', 'sqrt', 'log2'],
				'classifier__max_depth': range(3, 10),
				'classifier__min_samples_split': range(2, 10),
				'classifier__min_samples_leaf': range(1, 10),
				'classifier__bootstrap': [True, False],
			}
		elif model_type == 'xgboost':
			param_grid = {
				'classifier__n_estimators': stats.randint(200, 2000),
				'classifier__learning_rate': np.random.uniform(1e-3, 0.2, 100),
				'classifier__subsample': np.random.uniform(0.9, 1, 100),
				'classifier__max_depth': stats.randint(3, 10),
				'classifier__colsample_bytree': np.random.uniform(0.7, 1, 100),
				'classifier__min_child_weight': stats.randint(1, 5),
				'classifier__gamma': np.random.uniform(0.5, 5, 100),
			}
		elif model_type == 'logistic':
			param_grid = {

			}
	return param_grid

def wrap_model(model, task_type='regression'):
	pipeline = []
	if task_type == 'classification':
		pipeline.append(('classifier', model))
	elif task_type == 'regression':
		pipeline.append(('regressor', model))
	pipe = Pipeline(pipeline)
	return pipe

def get_scoring(task_type='regression'):
	scoring = None
	if task_type == 'regression':
		scoring = ['r2', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error']
	elif task_type == 'classification':
		scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
	return scoring

def random_search(model, model_type, df, task_type='regression', refit='r2', verbose=False):
	years_present = len(df.index.get_level_values('year').unique())
	starting_year = STARTING_YEAR
	last_year = starting_year + years_present - 1
	train_year = starting_year + round(years_present * .7)
	validation_year = starting_year + round((train_year-starting_year) * .8)

	test_year = train_year + 1
	X = df[(starting_year <= df.index.get_level_values('year')) & (df.index.get_level_values('year') <= train_year)]
	mask = (starting_year <= X.index.get_level_values('year')) & (X.index.get_level_values('year') <= validation_year)
	validation_fold = list(map(lambda x: -1 if x else 0, list(mask)))
	ps = PredefinedSplit(validation_fold)

	random_search = RandomizedSearchCV(model, param_distributions=get_param_grid(model_type=model_type, task_type=task_type), cv=ps, n_iter=10, n_jobs=-1, verbose=10, refit=True, scoring=refit, random_state=SEED)

	if verbose:
		print('Fitting with Random Search...')
	X_train = X.drop('crime_count', axis=1)
	y_train = X['crime_count']
	if task_type == 'classification':
		y_train = y_train.apply(lambda x: 1 if x > 0.5 else 0).round(0).astype(int)
	random_search.fit(X_train, y_train)
	if verbose:
		print('Done fitting...')

	return random_search.best_estimator_

def predict_from_model(model, df, task_type='regression', verbose=False):
	scoring = get_scoring(task_type=task_type)

	starting_year = STARTING_YEAR
	years_present = len(df.index.get_level_values('year').unique())
	last_year = starting_year + years_present - 1
	train_year = starting_year + round(years_present * .7)
	validation_year = starting_year + round((train_year-starting_year) * .8)

	test_year = train_year + 1
	mask = (starting_year <= df.index.get_level_values('year')) & (df.index.get_level_values('year') <= train_year)

	y = df.loc[~mask, 'crime_count']
	if task_type == 'classification':
		y = y.apply(lambda x: 1 if x > 0.5 else 0).round(0).astype(int)
	y_pred = model.predict(df.loc[~mask, df.columns != 'crime_count'])

	scores = {}
	if verbose:
		print('Scores:')
	for metric in scoring:
		str_metric = metric[4:] if metric.startswith('neg_') else metric
		if metric == 'r2':
			scores[str_metric] = r2_score(y, y_pred)
		elif metric == 'max_error':
			scores[str_metric] = max_error(y, y_pred)
		elif metric == 'neg_mean_absolute_error':
			scores[str_metric] = mean_absolute_error(y, y_pred)
		elif metric == 'neg_mean_squared_error':
			scores[str_metric] = mean_squared_error(y, y_pred)
		elif metric == 'accuracy':
			scores[str_metric] = accuracy_score(y, y_pred)
		elif metric == 'f1':
			scores[str_metric] = f1_score(y, y_pred)
		elif metric == 'precision':
			scores[str_metric] = precision_score(y, y_pred)
		elif metric == 'recall':
			scores[str_metric] = recall_score(y, y_pred)
		elif metric == 'roc_auc':
			scores[str_metric] = roc_auc_score(y, y_pred)

		if verbose:
			print('\t' + str_metric + ': %.2f' % scores[str_metric])

	return scores, y, y_pred
