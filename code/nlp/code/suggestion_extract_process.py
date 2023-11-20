
import subprocess
import sys

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(sys.path)

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
from pyspark.sql.types import ArrayType
import re
from pyspark.sql.functions import explode, count

spacy_version = "3.7.2"  
thinc_version = "8.2.1" 
pydantic_version = "1.8.0" 

# Install the packages using pip
subprocess.check_call([sys.executable, "-m", "pip", "install", f"spacy=={spacy_version}"])
subprocess.check_call([sys.executable, "-m", "pip", "install", f"thinc=={thinc_version}"])
subprocess.check_call([sys.executable, "-m", "pip", "install", f"pydantic=={pydantic_version}"])
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
import spacy

logging.basicConfig(format='%(asctime)s,%(levelname)s,%(module)s,%(filename)s,%(lineno)d,%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def main():
    
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_dataset_path", type=str, help="Path of dataset in S3")    
    parser.add_argument("--col_name_for_filtering", type=str, help="Name of the column to filter")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
    logger.info(f"spark version = {spark.version}")
    
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

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Define schema for the UDF output
    movie_schema = StructType([
        StructField("movie_positions", ArrayType(ArrayType(StringType()))),
        StructField("movie_names", ArrayType(StringType()))
    ])

    # UDF to extract movie names
    @udf(movie_schema)
    def extract_movie_names_udf(text):
        doc = nlp(text)
        movie_positions = []
        movie_names = []

        for ent in doc.ents:
            if ent.label_ == "ORG" or ent.label_ == "PERSON":
                movie_positions.append([ent.start_char, ent.end_char])
                movie_names.append(ent.text)

        return (movie_positions, movie_names)

    # UDF to remove movie names
    @udf(StringType())
    def remove_movie_names_udf(text, movie_positions):
        # Reverse sort positions to avoid shifting positions
        if movie_positions:
            movie_positions = sorted([(int(start), int(end)) for start, end in movie_positions], key=lambda x: x[0], reverse=True)

            for start, end in movie_positions:
                text = text[:start] + ' ' + text[end:]

            return ' '.join(text.split())

        else:
            return text

    # UDF to extract movie names using regex
    @udf(ArrayType(StringType()))
    def extract_movie_names_regex_udf(text, movie_names):
        movie_name_pattern = r'(?:\"([^\"]+)\"|([A-Z][a-z]*(?:\s+(?:[a-z]+\s+)*[A-Z][a-z]*)*)(?: \(\d{4}\))?)'

        movie_matches = re.findall(movie_name_pattern, text)
        movies = [m for match in matches for m in match if m]
        return movie_names + movies
    
    def remove_stop_word_from_movie_names(suggestion_list):
        
        if suggestion_list and len(suggestion_list[0].split()) == 1 and suggestion_list[0].lower() in stop_words:
            return suggestion_list[1:]
        
        return suggestion_list

    # Applying the UDFs to the DataFrame
    df_with_suggestions = comments_filtered.withColumn("movie_data", extract_movie_names_udf(comments_filtered["body"]))
    df_removed_suggestions_names = df_with_suggestions.withColumn("body_no_movies", remove_movie_names_udf(comments["body"], df_with_movie_data["movie_data.movie_positions"]))
    df_final = df_removed_suggestions_names.withColumn("suggestion_list", extract_movie_names_regex_udf(df_removed_movie_names["body_no_movies"], df_removed_movie_names["movie_data.movie_names"]))
    
    df_final = df_final.select("subreddit", "author", "body", "parent_id", "id", "created_utc", "score", "controversiality", "suggestion_list")
    
    stop_words = set(["a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "there", "he", "she"])  # stop words
    
    df_final = df_final.withColumn("suggestion_lists", remove_stop_word_udf(df_final["suggestion_list"]))
    
    remove_stop_word_udf = udf(remove_stop_word_from_movie_names, ArrayType(StringType()))

    # Flatten the movie_names column
    df_flattened = df_final_sample.withColumn("suggestions", explode(col("suggestion_lists")))

    # Group by movie_name and count the occurrences
    df_suggestion_frequency = df_flattened.groupBy("suggestions").agg(count("*").alias("frequency"))

    df_suggestion_frequency.write.mode("overwrite").parquet("{s3_path}/nlp/")
    
    logger.info(f"all done...")
    
if __name__ == "__main__":
    main()
