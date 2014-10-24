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
    option_list = [
                   {'name':'aamc 10 biological science review'},
                   {'name':'aamc 10 physical science review'},
                   {'name':'aamc 10 review small group'},
                   {'name':'aamc 11 biological science review'},
                   {'name':'aamc 11 physical science review'},
                   {'name':'aamc 11 review small group'},
                   {'name':'aamc 7 biological science review'},
                   {'name':'aamc 7 physical science review'},
                   {'name':'aamc 8 biological science review'},
                   {'name':'aamc 8 physical science review'},
                   {'name':'aamc 9 biological science review'},
                   {'name':'aamc 9 physical science review'},
                   {'name':'aamc 9 review small group'},
                   {'name':'ac and dc circuits'},
                   {'name':'acids and bases'},
                   {'name':'analytical writing'},
                   {'name':'biochemistry'},
                   {'name':'calculus'},
                   {'name':'channel'},
                   {'name':'electrostatics and magnets'},
                   {'name':'geometry'},
                   {'name':'integrated reasoning'},
                   {'name':'mcat diagnostic review small group'},
                   {'name':'mcat fl 1 review small group'},
                   {'name':'molecular genetics'},
                   {'name':'msct workshop 1'},
                   {'name':'msct workshop 2'},
                   {'name':'msct workshop 3'},
                   {'name':'organic chemistry 1'},
                   {'name':'organic chemistry 2'},
                   {'name':'organic chemistry 3'},
                   {'name':'perceptual ability 1'},
                   {'name':'perceptual ability 2'},
                   {'name':'physics 1'},
                   {'name':'physics 2'},
                   {'name':'quant mastery a'},
                   {'name':'quant mastery b'},
                   {'name':'statistics'},
                   {'name':'test analysis and patterns workshop 1'},
                   {'name':'test analysis and patterns workshop 2'},
                   {'name':'test analysis and patterns workshop 3'},
                   {'name':'the kidney'},
                   {'name':'thermodynamics'},
                   {'name':'verbal'},
                   {'name':'verbal mastery'}
    ]    
    return render_template('index.html',
			   title='Clustering WebApp',
			   option_list = option_list)

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
    if date == None or date == "":	
        responses = survey_comments.find({'type':type})     # Constrain data here
    else:
        responses = survey_comments.find({'type':type,
	    				      'date':{'$gte': date}})     # Constrain data here 
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
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000, stop_words='english')
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
            temp_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000, stop_words='english')
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
			   time = total_time)
