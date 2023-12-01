
import sys
import os
import logging
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import sagemaker
from pyspark.sql.functions import col, concat_ws, length

# Logging setup
logging.basicConfig(format='%(asctime)s,%(levelname)s,%(module)s,%(filename)s,%(lineno)d,%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def main():
    # Build Spark session
    spark = (SparkSession.builder
             .appName("PySparkApp")
             .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.2")
             .config(
                    "fs.s3a.aws.credentials.provider",
                    "com.amazonaws.auth.ContainerCredentialsProvider")
             .getOrCreate())
    
    logger.info(f"Spark version: {spark.version}")

    # Downloading data from S3 into a Dataframe
    transformed_train_data = spark.read.parquet("s3a://project-group34/project/submissions/preprocessed_ML2/train/")
    transformed_test_data = spark.read.parquet("s3a://project-group34/project/submissions/preprocessed_ML2/test/")

    # Logistic Regression Classifier
    lr_classifier = LinearRegression(labelCol="score", featuresCol="scaled_features")

    # Pipeline
    pipeline = Pipeline(stages=[lr_classifier])

    # Fit the pipeline
    pipeline_fit = pipeline.fit(transformed_train_data)
    train_predictions = pipeline_fit.transform(transformed_train_data)
    test_predictions = pipeline_fit.transform(transformed_test_data)

    # Write predictions to S3
    train_predictions.write.parquet("s3a://project-group34/project/submissions/LinearRegression/train_pred/", mode="overwrite")    
    test_predictions.write.parquet("s3a://project-group34/project/submissions/LinearRegression/test_pred/", mode="overwrite")
    pipeline_fit.save("s3a://sk2224-project/project/submissions/LinearRegression/model3/")

    logger.info(f"all done...")

if __name__ == "__main__":
    main()
