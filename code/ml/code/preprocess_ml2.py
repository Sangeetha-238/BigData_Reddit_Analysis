
import sys
import os
import logging
import argparse
# Import pyspark and build Spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline, Model
import sagemaker
from pyspark.sql.functions import lower, regexp_replace, col, concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import length
from pyspark.ml.feature import StandardScaler


logging.basicConfig(format='%(asctime)s,%(levelname)s,%(module)s,%(filename)s,%(lineno)d,%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def main():
    
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_dataset_path", type=str, help="Path of dataset in S3")    
    args = parser.parse_args()

    spark = SparkSession.builder \
    .appName("Spark ML")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.3")\
    .getOrCreate()
    
    logger.info(f"Spark version: {spark.version}")
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().set(
        "mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter"
    )

    # Downloading the data from S3 into a Dataframe
    logger.info(f"going to read {args.s3_dataset_path}")
    rescaledData = spark.read.parquet(args.s3_dataset_path, header=True)
    
    
    train_data, test_data, val_data = rescaledData.randomSplit([0.8, 0.15, 0.05], seed=2224)
    
    # Print the number of records in each dataset
    logger.info("Number of training records: " + str(train_data.count()))
    logger.info("Number of testing records: " + str(test_data.count()))
    logger.info("Number of validation records: " + str(val_data.count()))
    
    stringIndexer_over_18 = StringIndexer(inputCol="over_18", outputCol="over_18_ix")
    stringIndexer_is_self = StringIndexer(inputCol="is_self", outputCol="is_self_ix")
    stringIndexer_is_video = StringIndexer(inputCol="is_video", outputCol="is_video_ix")
    stringIndexer_has_media = StringIndexer(inputCol="has_media", outputCol="has_media_ix")
    stringIndexer_subreddit = StringIndexer(inputCol="subreddit", outputCol="subreddit_ix")
    stringIndexer_emotion = StringIndexer(inputCol="emotion", outputCol="emotion_ix")
    onehot_over_18 = OneHotEncoder(inputCol="over_18_ix", outputCol="over_18_vec")
    onehot_is_self = OneHotEncoder(inputCol="is_self_ix", outputCol="is_self_vec")
    onehot_is_video = OneHotEncoder(inputCol="is_video_ix", outputCol="is_video_vec")
    onehot_has_media = OneHotEncoder(inputCol="has_media_ix", outputCol="has_media_vec")
    onehot_subreddit = OneHotEncoder(inputCol="subreddit_ix", outputCol="subreddit_vec")
    vectorAssembler_features = VectorAssembler(
                                    inputCols=["over_18_vec", "is_self_vec", "is_video_vec", 
                                               "has_media_vec", "subreddit_vec", "num_comments",
                                               "post_length", "emotion_ix","sentiment_score","hour_of_day", "day_of_week", 
                                               "day_of_month", "month", "year"],
                                    outputCol="combined_features")

    # Scale the features
    scaler = StandardScaler(inputCol="combined_features", outputCol="scaled_features", withStd=True, withMean=False)

    # Define the stages for the pipeline
    stages = [
        stringIndexer_subreddit, 
        stringIndexer_over_18, 
        stringIndexer_is_self, 
        stringIndexer_is_video, 
        stringIndexer_has_media,
        stringIndexer_emotion,
        onehot_over_18,
        onehot_is_self,
        onehot_is_video,
        onehot_has_media,
        onehot_subreddit,
        vectorAssembler_features,
        scaler
    ]

    # Define the pipeline without the classifier and evaluator
    pipeline = Pipeline(stages=stages)

    # Fit the preprocessing part of the pipeline
    pipeline_fit = pipeline.fit(train_data)

    # Transform the data
    transformed_train_data = pipeline_fit.transform(train_data)
    transformed_test_data = pipeline_fit.transform(test_data)
    
    transformed_train_data.write.parquet("s3a://project-group34/project/submissions/preprocessed_ML2/train/",mode="overwrite")
    transformed_test_data.write.parquet("s3a://project-group34/project/submissions/preprocessed_ML2/test/",mode="overwrite")
    
if __name__ == "__main__":
    main()
