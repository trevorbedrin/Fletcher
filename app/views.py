import sys
import time
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
from sklearn.decomposition import PCA
from flask import render_template
from app import app

def get_centroid(data):
    return data.sum(0) / len(data)

@app.route('/')
@app.route('/index')
def index():
    # Connect to MongoDB
    client = MongoClient()
    db = client.dsbc

    # Set Collection
    survey_comments = db.survey_comments

    session_list = []

    responses = sorted(survey_comments.distinct('type')) 

    for response in responses:
	session_list.append({'name': response})

    presenter_list = []

#    responses = sorted(survey_comments.distinct('presenter'))

#    for response in responses:
#        presenter_list.append({'name': response})

    return render_template('index.html',
			   title='Clustering WebApp',
			   session_list = session_list,
			   presenter_list = presenter_list)

@app.route('/cluster', methods=['GET', 'POST'])
def cluster():
    start_time = time.time()
    # Connect to MongoDB
    client = MongoClient()
    db = client.dsbc

    # Set Collection
    survey_comments = db.survey_comments

    # Build Data Set(s)
    negative_data = [] # Negative Sentiment Comments
    neutral_data = []  # Neutral Sentiment Comments
    positive_data = [] # Positive Sentiment Comments
    polar_data = []    # Positive + Negative Comments
    all_data = []      # All Comments
  
    # Get data
    type = flask.request.form['type']
    data = flask.request.form['data-set']	
    epsilon = float(flask.request.form['epsilon']) 
    samples = int(flask.request.form['samples'])
    date = flask.request.form['date'] 
    presenter = flask.request.form['presenter']

    params = {'type':type}

    if presenter != 'all':
	params['presenter'] = presenter

    if date != None and date != "":	
        params['date'] = {'$gte': date}
    
    responses = survey_comments.find(params)

    for response in responses:
        response_text = response['response']
        blob = TextBlob(response_text)
        response_text = [response_text]
    ## Comment this out to cluster entire comments instead of sentences ##
    #    response_text = [sentence.string for sentence in blob.sentences]
    ######################################################################
        response_code = response['code']
        response_type = response['type']
        response_sentiment = float(response['sentiment'])
        if response_sentiment < 0:
            negative_data.extend(response_text)
            polar_data.extend(response_text)
            all_data.extend(response_text)
        elif response_sentiment > 0:
            positive_data.extend(response_text)
            polar_data.extend(response_text)
            all_data.extend(response_text)
        else:
            neutral_data.extend(response_text)
            all_data.extend(response_text)

    if data == 'negative':
        data_set = negative_data
    elif data == 'positive':
        data_set = positive_data
    else:
        data_set = polar_data
   
    # Build Vector of TfIdf Values
    vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=1000, stop_words='english')
    X_one = vectorizer.fit_transform(data_set).toarray()
    try:
	pca = PCA(n_components=1000)
        new_X = pca.fit_transform(X_one.todense())
        X = new_X
    except:
        X = X_one

    try:
        # Build Model
        model_dbscan = DBSCAN( eps = epsilon, min_samples = samples, metric ='cosine', algorithm='brute' )

        # Extract Clusters
        clusters_dbscan = model_dbscan.fit_predict(X)    

        # Build Each Cluster as a List of Strings
        cluster_contents_dbscan = defaultdict(list)

        for i in range(0, len(data_set)):
            if clusters_dbscan[i] >= 0:
                cluster_contents_dbscan[clusters_dbscan[i]].append(data_set[i])    

        dict_list = []

        for i in range(0, len(cluster_contents_dbscan)):
            temp_data = np.array(data_set)[clusters_dbscan==i]
            temp_vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=1000, stop_words='english')
            temp_X = temp_vectorizer.fit_transform(temp_data)
            distance = metrics.pairwise.pairwise_distances(temp_X, Y=get_centroid(temp_X.toarray()).tolist())
            min_distance = np.min(distance)
            index = distance.tolist().index(min_distance)
            dict_list.append({'name': "Cluster%d" % (i+1),
                              'centroid': temp_data[index],
	          	      'count': len(temp_data),
	                      'list': temp_data})
    except:
        dict_list = [{'name': 'Error',
                      'centroid': sys.exc_info()[0],
		      'count': 0,
		      'list': ''}]
    
    total_time = "%.2f sec" % (time.time() - start_time)        
    return render_template('cluster.html',
                           title='Clusters',
                           clusters = dict_list,
		           type = type,
                           date = date,
			   comment_count = len(data_set),
			   epsilon = epsilon,
			   samples = samples,
			   time = total_time,
			   data_set = data,
			   presenter = presenter)
