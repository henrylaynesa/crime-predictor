from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, Polygon, LinearRing
import pygeohash as pgh

def get_bounding_box_coordinates(geohash):
	# Returns in upper left, upper right, lower right, lower left in that order
	latitude, longitude, latitude_delta, longitude_delta = pgh.decode_exactly(geohash)
	points = [Point(longitude - longitude_delta, latitude + latitude_delta),
			  Point(longitude + longitude_delta, latitude + latitude_delta),
			  Point(longitude + longitude_delta, latitude - latitude_delta),
			  Point(longitude - longitude_delta, latitude - latitude_delta)]
	polygon = []
	for point in points:
		polygon.append([geohash, point])
	return polygon

def get_polygon_from_geohash(geohash):
	latitude, longitude, latitude_delta, longitude_delta = pgh.decode_exactly(geohash)
	points = [(longitude - longitude_delta, latitude + latitude_delta),
			  (longitude + longitude_delta, latitude + latitude_delta),
			  (longitude + longitude_delta, latitude - latitude_delta),
			  (longitude - longitude_delta, latitude - latitude_delta)]
	return Polygon(LinearRing(points))

def geohash_df(gdf, latitude='LATITUDE', longitude='LONGITUDE', precision=6, col_name='GEOHASH'):
	gdf[col_name] = gdf.apply(lambda row: pgh.encode(row[latitude], row[longitude], precision=precision), axis=1)
	return gdf

def get_geohash_shape_df(gdf, col_name='GEOHASH'):
	hashes = gdf[col_name].unique()
	polygons = []
	for hash_ in hashes:
		polygons += get_bounding_box_coordinates(hash_)
	df = gpd.GeoDataFrame(polygons, columns=['shape_id', 'geometry'], geometry='geometry')

	df['geometry'] = df['geometry'].apply(lambda x: x.coords[0])

	df = df.groupby('shape_id')['geometry'].apply(lambda x: Polygon(x.tolist())).reset_index()

	df = gpd.GeoDataFrame(df, geometry='geometry')
	df.crs = {'init': 'epsg:4326'}
	return df

def plot_geohashes(gh_gdf, border_gdf):
	fig, ax = plt.subplots(figsize=(10, 10))
	ax.set_aspect('equal')
	gh_gdf.plot(ax=ax, alpha=0.1, edgecolor='gray')
	border_gdf.plot(ax=ax, alpha=0.2, edgecolor='black')

def get_neighbor(geohash, direction='right'):
	precision = len(geohash)
	latitude, longitude, latitude_delta, longitude_delta = pgh.decode_exactly(geohash)
	neighbor = None
	if direction == 'right':
		neighbor = pgh.encode(latitude, longitude + longitude_delta * 2, precision=precision)
	elif direction == 'left':
		neighbor = pgh.encode(latitude, longitude - longitude_delta * 2, precision=precision)
	elif direction == 'up':
		neighbor = pgh.encode(latitude + latitude_delta * 2, longitude, precision=precision)
	elif direction == 'down':
		neighbor = pgh.encode(latitude - latitude_delta * 2, longitude, precision=precision)
	elif direction == 'up-right':
		neighbor = pgh.encode(latitude + latitude_delta * 2, longitude + longitude_delta * 2, precision=precision)
	elif direction == 'up-left':
		neighbor = pgh.encode(latitude + latitude_delta * 2, longitude - longitude_delta * 2, precision=precision)
	elif direction == 'down-right':
		neighbor = pgh.encode(latitude - latitude_delta * 2, longitude + longitude_delta * 2, precision=precision)
	elif direction == 'down-left':
		neighbor = pgh.encode(latitude - latitude_delta * 2, longitude - longitude_delta * 2, precision=precision)
	return neighbor

def get_neighbors(geohash, degree=1):
	geohashes = []
	# get to left side
	curr_geohash = geohash
	for i in range(degree):
		curr_geohash = get_neighbor(curr_geohash, direction='left')
	geohashes.append(curr_geohash)

	# traverse left side
	for i in range(degree):
		curr_geohash = get_neighbor(curr_geohash, direction='up')
		geohashes.append(curr_geohash)

	# traverse upper side
	for i in range(degree*2):
		curr_geohash = get_neighbor(curr_geohash, direction='right')
		geohashes.append(curr_geohash)

	# traverse right side
	for i in range(degree*2):
		curr_geohash = get_neighbor(curr_geohash, direction='down')
		geohashes.append(curr_geohash)

	# traverse lower side
	for i in range(degree*2):
		curr_geohash = get_neighbor(curr_geohash, direction='left')
		geohashes.append(curr_geohash)

	# traverse lower side
	for i in range(degree-1):
		curr_geohash = get_neighbor(curr_geohash, direction='up')
		geohashes.append(curr_geohash)

	return set(geohashes)

def complete_geohashes(gdf, col_name='GEOHASH'):
	hashes = list(gdf[col_name].unique())
	hashes_df = pd.DataFrame(hashes, columns=[col_name])
	mapped_hashes = pd.DataFrame(list(map(pgh.decode_exactly, hashes)),
								 columns=['latitude', 'longitude', 'latitude_delta', 'longitude_delta'])
	mapped_hashes = pd.concat([hashes_df, mapped_hashes], axis=1)
	new_cells = []
	for i, j in mapped_hashes.sort_values('latitude').groupby('latitude'):
		cell_leftmost = j[j['longitude'] == j['longitude'].min()]
		cell_rightmost = j[j['longitude'] == j['longitude'].max()]
		curr_cell = cell_leftmost['geohash'].iloc[0]
		while curr_cell != cell_rightmost['geohash'].iloc[0]:
			if len(mapped_hashes[mapped_hashes['geohash'] == curr_cell]) == 0:
				new_cells.append(curr_cell)
			curr_cell = get_neighbor(curr_cell, direction='right')
	hashes += new_cells
	return hashes

def get_geohashes_from_border(border_polygon, precision=7):
	x, y = border_polygon.exterior.coords.xy

	min_long = np.min(x)
	max_long = np.max(x)
	min_lat = np.min(y)
	max_lat = np.max(y)

	geohashes = []

	starting_latitude, starting_longitude, latitude_delta, longitude_delta = pgh.decode_exactly(
		pgh.encode(min_lat, min_long, precision=precision))
	ending_latitude, ending_longitude, latitude_delta, longitude_delta = pgh.decode_exactly(
		pgh.encode(max_lat, max_long, precision=precision))

	for long_ in np.arange(starting_longitude, ending_longitude + longitude_delta * 2, longitude_delta * 2):
		for lat in np.arange(starting_latitude, ending_latitude + latitude_delta * 2, latitude_delta * 2):
			curr_geohash = pgh.encode(lat, long_, precision=precision)
			if border_polygon.intersects(get_polygon_from_geohash(curr_geohash)):
				geohashes.append(curr_geohash)

	return geohashes