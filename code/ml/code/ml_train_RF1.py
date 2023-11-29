
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


logging.basicConfig(format='%(asctime)s,%(levelname)s,%(module)s,%(filename)s,%(lineno)d,%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def main():

    spark = (SparkSession.builder\
             .appName("PySparkApp")\
             .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.2")\
             .config(
                    "fs.s3a.aws.credentials.provider",
                    "com.amazonaws.auth.ContainerCredentialsProvider",
                )\
    .getOrCreate()
)
    
    logger.info(f"Spark version: {spark.version}")
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().set(
        "mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter"
    )

    # Downloading the data from S3 into a Dataframe
    transformed_train_data = spark.read.parquet("s3a://project-group34/project/submissions/preprocessed_ML/train/")
    transformed_test_data = spark.read.parquet("s3a://project-group34/project/submissions/preprocessed_ML/test/")
    
    # RandomForestClassifier without hyperparameter tuning
    rf_classifier = RandomForestClassifier(labelCol="subreddit_ix", 
                                           featuresCol="combined_features", 
                                           numTrees=100, 
                                           maxDepth=10,
                                           maxBins=64)
    labelConverter = IndexToString(inputCol="prediction", 
                               outputCol="predictedSubreddit", 
                               labels=["anime", "movie"])

    # Add the best model and labelConverter to the pipeline
    pipeline = Pipeline(stages=[rf_classifier, 
                                labelConverter])
    
    # Fit the entire pipeline
    pipeline_fit = pipeline.fit(transformed_train_data)
    train_predictions = pipeline_fit.transform(transformed_train_data)

    # Transform the data with the best model
    test_predictions = pipeline_fit.transform(transformed_test_data)
    
    train_predictions.write.parquet("s3a://project-group34/project/submissions/RandomForest/numTrees=100_maxDepth=10_maxBins=64/train_pred/", mode="overwrite")    
    test_predictions.write.parquet("s3a://project-group34/project/submissions/RandomForest/numTrees=100_maxDepth=10_maxBins=64/test_pred/", mode="overwrite")
    pipeline_fit.save("s3a://project-group34/project/submissions/RandomForest/numTrees=100_maxDepth=10_maxBins=64/model/")
    
    logger.info(f"all done...")
    
if __name__ == "__main__":
    main()
