from pymongo import MongoClient
import csv
from textblob import TextBlob
from textblob.en.sentiments import NaiveBayesAnalyzer

# Connect to MongoDB
client = MongoClient()
db = client.dsbc
db.collection_names()

# Set Collections
# survey_results = db.survey_results
survey_comments = db.survey_comments

# Build survey_comments collection

with open('survey_comments.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    print csv_reader.next()

    for row in csv_reader:
        row_data = {
        	'code': row[2], 
         	'type': row[3],
         	'date': row[6],
         	'question': row[14],
         	'response': row[15],
         	'presenter': "%s %s" % (row[8], row[9])
        } 
                        
        survey_comments.insert(row_data)
        
# Build Sentiment Analyzer and categorize comments.  Update Mongo with Sentiment scores

analyzer = NaiveBayesAnalyzer()
responses = survey_comments.find({"sentiment": {"$exists": False}})
for response in responses:
    response_id = response['_id']
    response_text = response['response']
    blob = TextBlob(response_text, classifier=analyzer)
    sentiment = blob.sentiment.polarity
    survey_comments.update({"_id": response_id},{"$set":{"sentiment": sentiment}},True)
    
    
