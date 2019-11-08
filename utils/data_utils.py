import numpy as np
import pandas as pd
import geopandas as gpd
import sys
sys.path.append('../source/')
sys.path.append('../utils/')
from config import *
from geo_utils import *
from datetime import date, timedelta
import time

def weeks_for_year(year):
	last_week = date(year, 12, 28)
	return last_week.isocalendar()[1]

def get_base_df(key):
	df = pd.read_csv(CONFIG_CSV[key])
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

	df = df.rename(columns={CONFIG_INDEX[key]['year']: 'year', CONFIG_INDEX[key]['week']: 'week', CONFIG_INDEX[key]['geohash']: 'geohash', CONFIG_TARGET[key]: 'crime_count_w-0'})
	df['year'] = pd.to_numeric(df['year'])
	df['week'] = pd.to_numeric(df['week'])
	df['crime_count_w-0'] = pd.to_numeric(df['crime_count_w-0'])

	df = df[df['year'] >= STARTING_YEAR]
	return df.drop(CONFIG_PREDROP[key], axis=1)

def get_prepared_df(key, prev_week_features, prev_weeks=1, neighbors=0):
	df = pd.read_csv(CONFIG_PREPARED_DF[key])
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

	temp_df = df.copy()

	df = df.loc[:, ~df.columns.str.contains('_w-')]
	for i in range(prev_weeks):
		df = pd.concat([df, temp_df.loc[:, temp_df.columns.str.contains('_w-'+str(i))]], axis=1)

	df = df.loc[:, ~df.columns.str.contains('_n-')]
	for i in range(neighbors):
		df = pd.concat([df, temp_df.loc[:, temp_df.columns.str.contains('_n-'+str(i))]], axis=1)
	return df

def get_previous_n_weeks(df, ind, n=1):
	year = ind[0]
	week = ind[1]
	if week == 1:
		year -= 1
	weeks_in_a_year = weeks_for_year(year)
	week = (week - n - 1) % weeks_in_a_year + 1
	return year, week, ind[2]


def get_spatial_data(df, row, ind, neighbors=0):
	for i in range(neighbors):
		neighbor_crime = 0
		neighbors_found = 0
		for neighbor in get_neighbors(ind[2], degree=i+1):
			new_ind = (ind[0], ind[1], neighbor)
			if new_ind in df.index:
				neighbor_crime += df.loc[new_ind, 'crime_count_w-0']
				neighbors_found += 1
		average_crime = 0
		if neighbors_found != 0:
			average_crime = neighbor_crime / neighbors_found
		ser = pd.Series([average_crime], index=['crime_count_n-' + str(i+1)])
		row = row.append(ser)
	
	row = pd.Series(list(ind), index=['year', 'week', 'geohash']).append(row)
	return row

def prepare_data(df, prev_week_features, prev_weeks=1, neighbors=0, verbose=False):
	# time shift
	for i in range(prev_weeks):
		df['crime_count_w-'+str(i+1)] = pd.Series(np.nan, index=df.index)
	for geohash in df['geohash'].unique():
	    temp = df.loc[df['geohash'] == geohash, ['year', 'week', 'crime_count_w-0']].sort_values(['year', 'week'])
	    for i in range(prev_weeks):
	    	df.loc[temp.index, 'crime_count_w-'+str(i+1)] = temp['crime_count_w-0'].shift(i+1)
	    df.loc[temp.index, 'crime_count'] = temp['crime_count_w-0'].shift(-1)
	df = df.loc[~df.isnull().any(axis=1)]

	# spatial
	df = df.set_index(['year', 'week', 'geohash'])
	df = df.loc[~df.index.duplicated(keep='first')]
	df = df.sort_index()
	new_list = []

	curr_year = -1
	curr_week = -1

	for i, r in df.iterrows():
		if verbose:
			if i[0] != curr_year:
				curr_year = i[0]
				print('Processing Year %s' % curr_year)
			if i[1] != curr_week:
				curr_week = i[1]
				print('Processing Week %s' % curr_week)
		new_list.append(get_spatial_data(df, r, i, neighbors=neighbors))
	return pd.DataFrame(new_list)
