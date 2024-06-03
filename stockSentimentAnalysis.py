#!/usr/bin/env python
# coding: utf-8



from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col,udf
from pyspark.sql.functions import to_timestamp, to_date
from pyspark.sql.functions import lit
from pyspark.sql.types import (StructType,StructField,StringType,IntegerType,DoubleType,DateType)
import boto3
from rake_nltk import Rake

spark = SparkSession.builder.appName('TwintApp2').getOrCreate()


#save as dataframe
df=spark.read.json('/Users/bryankennethbarrion/Documents/Twitter/twint/', encoding='utf-8')

def ascii_ignore(x):
    return x.encode('ascii', 'ignore').decode('ascii')

ascii_udf = udf(ascii_ignore)
#drop row with no tweet
df=df.na.drop(subset=['tweet'])
#drop special characters
df=df.withColumn("tweet", ascii_udf('tweet'))
df=df.withColumn('created_at', to_date(df['date']))
df=df.filter(df['retweet']==False)



# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def checkKeyWords(extract_tweet_only):
    r=Rake()
    r=Rake(min_length=1,max_length=2)
    r.extract_keywords_from_text(extract_tweet_only)
    keywords=r.get_ranked_phrases()
    while len(keywords) < 10:
        keywords.append(None)
    return keywords[0:10]


session = boto3.Session(region_name='ap-southeast-1')
results=df.select(['created_at','tweet','username','likes_count','replies_count','retweets_count']).collect()
extract_tweet_only=df.select(['tweet']).collect()


data=[]
batchSize = 25
batchData = list(divide_chunks(extract_tweet_only, batchSize))
count=0
for bdata in batchData:
    keyWords=[]
    sentimentData=[]
    print(len(bdata))
    for i in range(len(bdata)):
        tweet=bdata[i][0]
        keyWords.append(checkKeyWords(tweet))
        sentimentData.append(tweet)

    comprehend_client = session.client('comprehend')
    detect_sentiment=comprehend_client.batch_detect_sentiment(
        TextList=sentimentData,
        LanguageCode='en'
    )

    for res in detect_sentiment['ResultList']:
        sentiment=res['Sentiment']
        pos=res['SentimentScore']['Positive']
        neg=res['SentimentScore']['Negative']
        neu=res['SentimentScore']['Neutral']
        mix=res['SentimentScore']['Mixed']
        result=results.pop(0)
        keyWord=keyWords.pop(0)
        #resultList.append((sentiment,posSentiment,negSentiment,neutralSentiment,mixedSentiment))
        data.append((result['created_at'],result['tweet'],
            result['username'],result['likes_count'],result['replies_count'],
            result['retweets_count'],sentiment,pos,neg,neu,mix,
            keyWord[0],keyWord[1],keyWord[2],
            keyWord[3],keyWord[4],keyWord[5],
            keyWord[6],keyWord[7],keyWord[8],keyWord[9]))



schema = StructType([     StructField("date",DateType(),True),     StructField("tweet",StringType(),True),     StructField("username",StringType(),True),     StructField("likes_count", IntegerType(), True),     StructField("replies_count", IntegerType(), True),     StructField("retweets_count", IntegerType(), True),     StructField("sentiment",StringType(),True),     StructField("positive", DoubleType(), True),     StructField("negative", DoubleType(), True),     StructField("neutral", DoubleType(), True),     StructField("mixed", DoubleType(), True),     StructField("key_phrase1", StringType(), True),     StructField("key_phrase2", StringType(), True),     StructField("key_phrase3", StringType(), True),     StructField("key_phrase4", StringType(), True),     StructField("key_phrase5", StringType(), True),     StructField("key_phrase6", StringType(), True),     StructField("key_phrase7", StringType(), True),     StructField("key_phrase8", StringType(), True),     StructField("key_phrase9", StringType(), True),     StructField("key_phrase10", StringType(), True),   ])

df = spark.createDataFrame(data=data, schema=schema)

df.write.format("parquet").option('header',True).mode('overwrite').save('TwitterOutputParquet')