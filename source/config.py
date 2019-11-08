CRIME_TYPES = set(['BURGLARY', 'THEFT F/AUTO', 'MOTOR VEHICLE THEFT'])
MODELS_SUPPORTED = set(['random_forest', 'xgboost', 'logistic'])
STARTING_YEAR = 2010

CONFIG_CSV = {
	'motor': '../data/input/motor.csv',
	'burglary': '../data/input/burglary.csv',
	'auto': '../data/input/auto.csv'
}

CONFIG_INDEX = {
	'motor': {'year': 'start_year', 'week': 'start_week', 'geohash': 'geohash'},
	'burglary': {'year': 'start_year', 'week': 'start_week', 'geohash': 'geohash'},
	'auto': {'year': 'year', 'week': 'week', 'geohash': 'geohash'}
}

CONFIG_TASKS = {
	'motor': 'MOTOR VEHICLE THEFT', 
	'burglary': 'BURGLARY',
	'auto': 'THEFT F/AUTO'
}

CONFIG_ASSIGNMENTS = {
	'MOTOR VEHICLE THEFT': 'motor',
	'BURGLARY': 'burglary',
	'THEFT F/AUTO': 'auto'
}

CONFIG_TARGET = {
	'motor': 'crime_count', 
	'burglary': 'crime_count',
	'auto': 'crime_count'
}

CONFIG_PREDROP = {
	'motor': [],
	'burglary': [],
	'auto': ['month', 'date', 'target_crime_count', 'geohash_shape']
}

CONFIG_PREVIOUS_WEEK_FEATURES = {
	'motor': [CONFIG_TARGET['motor']],
	'burglary': [CONFIG_TARGET['burglary']],
	'auto': [CONFIG_TARGET['auto']],
}

CONFIG_PREPARED_DF = {
	'motor': '../data/input/motor.csv',
	'burglary': '../data/input/burglary.csv',
	'auto': '../data/input/auto.csv'
}

CONFIG_NON_NUMERICAL_FEATURES = {
	'motor': [],
	'burglary': [],
	'auto': []
}

def CONFIG_LOAD_PICKLE(key, model_type='random_forest', task_type='regression'):
	if key == 'motor':
		return '../pickle/input/motor_' + task_type + '_' + model_type + '.pickle'
	elif key == 'burglary':
		return '../pickle/input/burglary_' + task_type + '_' + model_type + '.pickle'
	elif key == 'auto':
		return '../pickle/input/auto_' + task_type + '_' + model_type + '.pickle'

def CONFIG_SAVE_PICKLE(key, model_type='random_forest', task_type='regression'):
	if key == 'motor':
		return '../pickle/output/motor_' + task_type + '_' + model_type + '.pickle'
	elif key == 'burglary':
		return '../pickle/output/burglary_' + task_type + '_' + model_type + '.pickle'
	elif key == 'auto':
		return '../pickle/output/auto_' + task_type + '_' + model_type + '.pickle'