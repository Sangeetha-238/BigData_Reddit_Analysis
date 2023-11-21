
import subprocess
import sys

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(sys.path)

subprocess.check_call([sys.executable, "-m", "pip", "install", "spark-nlp"])

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
    
    # DATA CLEANING
    comments_filtered = df_filtered.filter((df.body != '[deleted]') & (df.author != '[deleted]'))
    comments_filtered_movies = comments_filtered.where(col("subreddit").isin("MovieSuggestions"))

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

    # Use a pretrained embeddings model, for example, BERT
    embeddings = BertEmbeddings.pretrained("bert_base_cased", "en") \
        .setInputCols(["sentence", "token"]) \
        .setOutputCol("embeddings")

    ner_model = NerDLModel.pretrained("ner_dl_bert", "en") \
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
    model = nlp_pipeline.fit(comments_filtered_movies)
    result = model.transform(comments_filtered_movies)
    
    print("NLP Pipeline Ran Succesfully!")

    # Define a UDF to filter and extract movie names
    def extract_movies(chunks):
        movie_names = [chunk.result for chunk in chunks if chunk.metadata['entity'] in ['PERSON', 'ORG']]
        return movie_names

    extract_movie_names_udf = udf(extract_movies, ArrayType(StringType()))

    # Apply the UDF to the DataFrame
    movies_df = result.withColumn("movie_names", extract_movie_names_udf(F.col("ner_chunk")))


    @udf(StringType())
    def remove_movie_names_udf(text, movie_names):
        if movie_names:
            for name in movie_names:
                text = text.replace(name, ' ')
            return ' '.join(text.split())
        else:
            return text

    # UDF to extract movie names using regex
    @udf(ArrayType(StringType()))
    def extract_movie_names_regex_udf(text, movie_names):
        movie_name_pattern = r'(?:\"([^\"]+)\"|([A-Z][a-z]*(?:\s+(?:[a-z]+\s+)*[A-Z][a-z]*)*)(?: \(\d{4}\))?)'

        movie_matches = re.findall(movie_name_pattern, text)
        movies = [match[0] or match[1] or match[2] for match in movie_matches]
        return movie_names + movies
    
    def remove_stop_word_from_movie_names(movies):
        if movies and len(movies[0].split()) == 1 and movies[0].lower() in eng_stopwords:
            return movies[1:]
        return movies

    # Remove movie names from the 'body' text
    df_removed_movie_names = movies_df.withColumn("body_no_movies", remove_movie_names_udf(movies_df["body"], movies_df["movie_names"]))

    # The regex method to supplement the NER extraction
    df_final = df_removed_movie_names.withColumn("additional_movie_names", extract_movie_names_regex_udf(df_removed_movie_names["body_no_movies"], df_removed_movie_names["movie_names"]))

    df_final = df_final.select("subreddit", "author", "body", "parent_id", "id", "created_utc", "score", "controversiality", "additional_movie_names")
    
    print("Movie Names Extracted")
    
    remove_stop_word_udf = udf(remove_stop_word_from_movie_names, ArrayType(StringType()))

    df_final = df_final.withColumn("movie_names_final", remove_stop_word_udf(df_final["additional_movie_names"]))
    
    # Flatten the movie_names column
    df_flattened = df_final.withColumn("movie_name", explode(col("movie_names_final")))

    # Group by movie_name and count the occurrences
    df_frequency = df_flattened.groupBy("movie_name").agg(count("*").alias("frequency"))
    
    print("Aggregation Done!")
    
    # Sort the DataFrame by frequency in descending order and take the top 1000
    df_top_1000_movies = df_frequency.orderBy(desc("frequency")).limit(1000)
    
    # df_top_1000_movies_pd = df_top_1000_movies.toPandas()
    
    bucket = "project-group34"
    output_prefix_data_comments = "project/comments/"
    s3_path = f"s3a://{bucket}/{output_prefix_data_comments}" + args.col_name_for_filtering + "/"
    
    print(f"Writing to {s3_path}")
    df_top_1000_movies.write.parquet(s3_path)
    # Save the DataFrame in CSV format to S3
    #df_top_1000_movies.write.option("header", "true").mode("overwrite").csv(s3_path)
    print(f"Finished writing to {s3_path}")

    # df_top_1000_movies.to_csv(f"{s3_path}/movie_suggestions/{csv_name}")
    
    logger.info(f"all done...")
    
if __name__ == "__main__":
    main()
