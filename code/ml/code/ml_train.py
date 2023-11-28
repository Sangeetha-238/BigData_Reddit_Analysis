
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
    
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_dataset_path_train", type=str, help="Path of train dataset in S3")    
    parser.add_argument("--s3_dataset_path_test", type=str, help="Path of test dataset in S3")
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
    transformed_train_data = spark.read.parquet(args.s3_dataset_path_train, header=True)
    transformed_test_data = spark.read.parquet(args.s3_dataset_path_test, header=True)
    
    # RandomForestClassifier without hyperparameter tuning
    rf_classifier = RandomForestClassifier(labelCol="subreddit_ix", featuresCol="combined_features", numTrees=30)

    # Hyperparameter tuning using CrossValidator
    param_grid_rf = ParamGridBuilder() \
        .addGrid(rf_classifier.numTrees, [50, 100]) \
        .addGrid(rf_classifier.maxDepth, [50, 100]) \
        .addGrid(rf_classifier.maxBins, [64, 128]) \
        .build()

    evaluator_rf = BinaryClassificationEvaluator(labelCol="subreddit_ix")

    crossval_rf = CrossValidator(estimator=rf_classifier,
                                 estimatorParamMaps=param_grid_rf,
                                 evaluator=evaluator_rf,
                                 numFolds=2)

    # Fit the CrossValidator to find the best model
    cv_model_rf = crossval_rf.fit(transformed_train_data)

    # Get the best model from the CrossValidator
    best_rf_model = cv_model_rf.bestModel

    # Add the best model and labelConverter to the pipeline
    pipeline_with_best_model = Pipeline(stages=stages + [best_rf_model, labelConverter])

    # Fit the entire pipeline
    pipeline_fit_with_best_model = pipeline_with_best_model.fit(train_data)
    train_predictions = pipeline_fit_with_best_model.transform(train_data)

    # Transform the data with the best model
    test_predictions = pipeline_fit_with_best_model.transform(test_data)
    
    train_predictions.write.parquet("s3a://project-group34/project/submissions/train_pred_RF/", mode="overwrite")    
    test_predictions.write.parquet("s3a://project-group34/project/submissions/test_pred_RF/", mode="overwrite")
    
    logger.info(f"all done...")
    
if __name__ == "__main__":
    main()
