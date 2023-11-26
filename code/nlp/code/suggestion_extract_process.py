
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
from sparknlp.base import Finisher, DocumentAssembler
from sparknlp.annotator import SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel, NerConverter
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
    parser.add_argument("--col_name_for_filtering", type=str, help="Name of the column to filter")
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
    logger.info(f"going to read {args.s3_dataset_path} for r/{args.col_name_for_filtering}")
    df = spark.read.parquet(args.s3_dataset_path, header=True)
    vals = [args.col_name_for_filtering]
    df_filtered = df.where(col("subreddit").isin(vals))
    logger.info(f"finished reading files...")
    logger.info(f"Number of rows in data: {df_filtered.count()}")
    
    # DATA CLEANING
    df_filtered = df_filtered.filter((df.body != '[deleted]') & (df.author != '[deleted]'))

    # Define the pipeline stages
    document_assembler = DocumentAssembler() \
        .setInputCol("body") \
        .setOutputCol("document")

    sentence_detector = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    tokenizer = Tokenizer() \
        .setInputCols(["sentence"]) \
        .setOutputCol("token")

    # Use GloVe embeddings
    embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en") \
        .setInputCols(["sentence", "token"]) \
        .setOutputCol("embeddings")

    # Use a lighter NER model
    ner_model = NerDLModel.pretrained("ner_dl", "en") \
        .setInputCols(["sentence", "token", "embeddings"]) \
        .setOutputCol("ner")

    ner_converter = NerConverter() \
        .setInputCols(["sentence", "token", "ner"]) \
        .setOutputCol("ner_chunk")

    # Build the pipeline
    nlp_pipeline = Pipeline(stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter
    ])

    # Apply the pipeline to your DataFrame
    model = nlp_pipeline.fit(df_filtered)
    result = model.transform(df_filtered)

    print("NLP Pipeline Ran Succesfully!")
                
    result = result.select("subreddit", "author", "body", "parent_id", "id", "created_utc", "score", "controversiality", "ner_chunk")

    # Define a UDF to filter and extract anime names
    def extract_anime(chunks):
        anime_names = [chunk.result for chunk in chunks if chunk.metadata['entity'] in ['PERSON', 'ORG']]
        return anime_names

    extract_anime_names_udf = udf(extract_anime, ArrayType(StringType()))

    # Apply the UDF to the DataFrame
    anime_df = result.withColumn("movie_names", extract_anime_names_udf(F.col("ner_chunk")))
    
    anime_df.write.parquet("s3a://project-group34/project/suggestions/all_anime/ner/", mode="overwrite")
    
    logger.info(f"all done...")
    
if __name__ == "__main__":
    main()
