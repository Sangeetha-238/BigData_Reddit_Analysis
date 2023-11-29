
import os
import logging
import argparse

# Import pyspark and build Spark session
from pyspark.sql.functions import *
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType
import re
from pyspark.sql.functions import explode, count
import sagemaker
from pyspark.sql.functions import lower, regexp_replace, col, concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from pyspark.ml import Pipeline

from pyspark.sql.functions import desc

import nltk
nltk.download('stopwords')
eng_stopwords = nltk.corpus.stopwords.words('english')

logging.basicConfig(format='%(asctime)s,%(levelname)s,%(module)s,%(filename)s,%(lineno)d,%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def main():
    
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_dataset_path", type=str, help="Path of dataset in S3")    
    args = parser.parse_args()

    spark = SparkSession.builder \
    .appName("Spark NLP")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.3")\
    .getOrCreate()
    
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"sparknlp version: {sparknlp.version()}")
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().set(
        "mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter"
    )

    # Downloading the data from S3 into a Dataframe
    logger.info(f"going to read {args.s3_dataset_path}")
    df = spark.read.parquet(args.s3_dataset_path, header=True)

    documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

    # Paths to the models
    # tfhub_use_path = "../../../cache_pretrained/tfhub_use_en_2.4.0_2.4_1587136330099/"
    # sentimentdl_use_twitter_path = "../../../cache_pretrained/sentimentdl_use_twitter_en_2.7.1_2.4_1610983524713/"
    # sentiment_emotion = "../../../cache_pretrained/cache_pretrained/classifierdl_use_emotion_en_2.7.1_2.4_1610190563302/"

    # Load models from local path
    use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
             .setInputCols(["document"])\
             .setOutputCol("sentence_embeddings")

    # sentimentdl = SentimentDLModel.pretrained(name="sentimentdl_use_twitter", lang="en")\
    #                  .setInputCols(["sentence_embeddings"])\
    #                  .setOutputCol("sentiment")

    sentimentdl1 = ClassifierDLModel.pretrained(name="classifierdl_use_emotion")\
        .setInputCols(["sentence_embeddings"])\
        .setOutputCol("sentiment_emotion")

    nlpPipeline = Pipeline(
          stages = [
              documentAssembler,
              use,
              sentimentdl1
              #sentimentdl
          ])

    # Apply the pipeline to your DataFrame
    model = nlpPipeline.fit(df)
    result = model.transform(df)
    
    result.write.parquet("s3a://project-group34/project/submissions/emotion_extracted/", mode="overwrite")
    
    logger.info(f"all done...")
    
if __name__ == "__main__":
    main()
