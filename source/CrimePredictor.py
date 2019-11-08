import sys
import os
sys.path.append('../utils/')
from config import *
from data_utils import *
from model_utils import *
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import xgboost as xgb


class CrimePredictor:

	def reset(self):
		self.df = None
		self.model = None
		self.y = None
		self.y_pred = None
		self.__config_key = CONFIG_ASSIGNMENTS[self.crime_type]


	def __init__(self, crime_type, model_type='random_forest', prev_weeks=1, neighbors=0, task_type='regression'):
		assert crime_type in CRIME_TYPES
		assert model_type in MODELS_SUPPORTED

		self.crime_type = crime_type
		self.model_type = model_type
		self.prev_weeks = prev_weeks
		self.neighbors = neighbors
		self.task_type = task_type
		self.reset()


	def load_data(self, force=False, verbose=False):
		prev_week_features = CONFIG_PREVIOUS_WEEK_FEATURES[self.__config_key]

		if force:
			base_df = get_base_df(self.__config_key)
			base_df = prepare_data(base_df, prev_week_features, prev_weeks=self.prev_weeks, neighbors=self.neighbors, verbose=verbose)
			base_df.to_csv(CONFIG_PREPARED_DF[self.__config_key].replace('input', 'output'), index=False)
		else:
			if verbose:
				print('Fetching prepared data...')
			base_df = get_prepared_df(self.__config_key, prev_week_features, prev_weeks=self.prev_weeks, neighbors=self.neighbors)

		self.df = base_df
		return self.df


	def preprocess(self, onehot=True, minmax_scale=False, normalize=False):
		non_numerical_features = CONFIG_NON_NUMERICAL_FEATURES[self.__config_key]

		df = self.df
		df = df.set_index(['year', 'week', 'geohash'])

		if onehot:
			df = one_hot_df(df, non_numerical_features)

		if minmax_scale:
			df = min_max_scaler_df(df, df.drop('crime_count', axis=1).columns)

		if normalize:
			df = normalize_df(df, df.drop('crime_count', axis=1).columns)

		self.df = df

		return self.df


	def train_test(self, force=False, verbose=False):
		if force:
			df = self.df

			model = wrap_model(get_model(self.model_type, task_type=self.task_type), task_type=self.task_type)

			if verbose:
				print('Training...')

			refit = ''
			if self.task_type == 'regression':
				refit = 'neg_mean_absolute_error'
			elif self.task_type == 'classification':
				refit = 'roc_auc'

			self.model = random_search(model, self.model_type, df, task_type=self.task_type, refit=refit, verbose=verbose)

			if verbose:
				print('Done!')
			pickle_out = open(CONFIG_SAVE_PICKLE(self.__config_key, model_type=self.model_type, task_type=self.task_type), 'wb')
			pickle.dump(self.model, pickle_out)
			pickle_out.close()
		else:
			if verbose:
				print('Using pre-trained model...')
			pickle_in = open(CONFIG_LOAD_PICKLE(self.__config_key, model_type=self.model_type, task_type=self.task_type), 'rb')
			self.model = pickle.load(pickle_in)
			pickle_in.close()
			
		scores, self.y, self.y_pred = predict_from_model(self.model, self.df, task_type=self.task_type, verbose=verbose)
		if verbose:
			print('Done!')
		
		return scores, self.y, self.y_pred

	def feature_importance(self):
		task_model = ''
		if self.task_type == 'classification':
			task_model = 'classifier'
		elif self.task_type == 'regression':
			task_model = 'regressor'
		else:
			return

		if self.model_type == 'random_forest':
			feature_importance = {}
			feature_importance_values = self.model.named_steps[task_model].feature_importances_
			columns = self.df.drop('crime_count', axis=1).columns
			for i in range(len(columns)):
				feature_importance[columns[i]] = feature_importance_values[i]
			pd.DataFrame({'Feature Importance': feature_importance}).sort_values(by='Feature Importance', ascending=False).plot(kind='barh', figsize=(10, 8))
			plt.grid()
			plt.gca().invert_yaxis()
		elif self.model_type == 'xgboost':
			fig, ax = plt.subplots(1, 1, figsize=(10, 8))
			xgb.plot_importance(self.model.named_steps[task_model], ax=ax)