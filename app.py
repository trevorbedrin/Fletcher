import flask
import numpy as np
import csv
import json
import unirest
from pprint import pprint
from collections import defaultdict
from pymongo import MongoClient
from textblob import TextBlob
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

def get_centroid(data):
    return data.sum(0) / len(data)

# Connect to MongoDB
client = MongoClient()
db = client.dsbc
db.collection_names()

# Set Collection
survey_comments = db.survey_comments

# Initialize App
app = flask.Flask("SurveyApp")

@app.route("/")
def viz_page():
    """ Homepage: serve our visualization page, awesome.html
    """
    with open("awesome.html", 'r') as viz_file:
        return viz_file.read()
        
# Handle request
@app.route("/cluster", methods=["POST"])
def cluster():
	"""  When A POST request with json data is made to this uri,
		 Read the example from the json, predict probability and
		 send it with a response
	"""
	# Build Data Set(s)
	negative_data = [] # Negative Sentiment Comments
	neutral_data = []  # Neutral Sentiment Comments
	positive_data = [] # Positive Sentiment Comments
	polar_data = []    # Positive + Negative Comments
	all_data = []      # All Comments
	

	# Get data
	data = flask.request.json
	type = data["type"]
	
	return type
# 	responses = survey_comments.find({'type':type})     # Constrain data here
# 	for response in responses:
# 		response_text = response['response']
# 		blob = TextBlob(response_text)
# 		response_text = [response_text]
# 	## Comment this out to cluster entire comments instead of sentences ##
# 	#    response_text = [sentence.string for sentence in blob.sentences]
# 	######################################################################
# 		response_code = response['code']
# 		response_type = response['type']
# 		response_sentiment = float(response['sentiment'])
# 		if response_sentiment < 0:
# 			negative_data.extend(response_text)
# 			polar_data.extend(response_text)
# 			all_data.extend(response_text)
# 		elif response_sentiment > 0:
# 			positive_data.extend(response_text)
# 			polar_data.extend(response_text)
# 			all_data.extend(response_text)
# 		else:
# 			neutral_data.extend(response_text)
# 			all_data.extend(response_text)
# 
# 	if data["data_set"] == 'negative':
# 		data_set = negative_data
# 	elif data["data_set"] == 'positive':
# 		data_set = positive_data
# 	else:
# 		data_set = polar_data
# 
# 	# Build Vector of TfIdf Values
# 	vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000, stop_words='english')
# 	X = vectorizer.fit_transform(data_set)
# 
# 	# Build Model
# 	model_dbscan = DBSCAN( eps = 0.8, min_samples = 5, metric ='cosine', algorithm='brute' )
# 
# 	# Extract Clusters
# 	clusters_dbscan = model_dbscan.fit_predict(X.toarray())    
# 
# 	# Build Each Cluster as a List of Strings
# 	cluster_contents_dbscan = defaultdict(list)
# 
# 	for i in range(0, len(data_set)):
# 		if clusters_dbscan[i] >= 0:
# 			cluster_contents_dbscan[clusters_dbscan[i]].append(data_set[i])    
# 
# 	tuple_dict = defaultdict(dict)
# 
# 	for i in range(0, len(cluster_contents_dbscan)):
# 		temp_data = np.array(data_set)[clusters_dbscan==i]
# 		temp_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000, stop_words='english')
# 		temp_X = temp_vectorizer.fit_transform(temp_data)
# 		distance = metrics.pairwise.pairwise_distances(temp_X, Y=get_centroid(temp_X.toarray()).tolist())
# 		min_distance = np.min(distance)
# 		index = distance.tolist().index(min_distance)
# 		tuple_dict["Cluster %d" % (i+1)] = temp_data[index]
# 
# 	return flask.jsonify(tuple_dict)

app.run(host='0.0.0.0', port=80)




