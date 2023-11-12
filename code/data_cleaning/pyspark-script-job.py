import argparse
import os
import logging
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, length, expr, to_timestamp, hour, dayofweek, dayofmonth, month, year
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql import functions as F

# Setup logging
logging.basicConfig(level=logging.INFO)

# Parse Inputs
parser = argparse.ArgumentParser()
parser.add_argument("--input_object_store_base_url")
parser.add_argument("--input_path")
parser.add_argument("--output_object_store_base_url")
parser.add_argument("--output_path")
args = parser.parse_args()

# Log arguments
logging.info(args.input_object_store_base_url)
logging.info(args.input_path)
logging.info(args.output_object_store_base_url)
logging.info(args.output_path)

# Paths for reading and writing data
input_complete_path = f"{args.input_object_store_base_url}{args.input_path}"
output_complete_path = f"{args.output_object_store_base_url}{args.output_path}"

# Initialize Spark session
spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# Log Spark version and input/output paths
logging.info(f"spark version = {spark.version}")
logging.info(input_complete_path)
logging.info(output_complete_path)

# Read data
logging.info(f"going to read {input_complete_path}")
submissions_df = spark.read.parquet(input_complete_path)

# Subreddit counts
subreddit_count_df = submissions_df.groupBy('subreddit').count()
subreddit_count_df.write.mode("overwrite").csv(f"{output_complete_path}/subreddit_count_eda.csv", header=True)

# Select specific columns
df = submissions_df.select("subreddit", "author", "title", "selftext",
                             "created_utc", "num_comments", "score", 
                             "over_18", "media", "pinned", "locked", 
                             "disable_comments", "domain", "hidden", 
                             "distinguished", "hide_score")

# Count missing values
missing_vals = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
# df_long = missing_vals_pd.melt(var_name='Column', value_name='Missing Values')
# df_long.to_csv(f"{output_complete_path}/num_missing_val_eda.csv", index=False)
missing_vals.write.mode("overwrite").csv(f"{output_complete_path}/num_missing_val_eda.csv", header=True)

# Feature engineering
df = df.withColumn('post_length', length(df.title) + length(df.selftext))
df = df.withColumn('created_utc', to_timestamp('created_utc'))
df = df.withColumn('hour_of_day', hour('created_utc'))
df = df.withColumn('day_of_week', dayofweek('created_utc'))
df = df.withColumn('day_of_week_str', expr("""
    CASE day_of_week 
        WHEN 1 THEN 'Sunday'
        WHEN 2 THEN 'Monday'
        WHEN 3 THEN 'Tuesday'
        WHEN 4 THEN 'Wednesday'
        WHEN 5 THEN 'Thursday'
        WHEN 6 THEN 'Friday'
        WHEN 7 THEN 'Saturday'
    END
"""))
df = df.withColumn('day_of_month', dayofmonth('created_utc'))
df = df.withColumn('month', month('created_utc'))
df = df.withColumn('year', year('created_utc'))
df = df.withColumn('has_media', col('media').isNotNull())
df = df.drop(*["media", "created_utc", "disable_comments", "distinguished"])

df_filtered = df.filter(df.subreddit.isin('movies', 'anime', 'television'))

# EDA on datetime features
df_datetime = df_filtered.groupBy(["subreddit", "day_of_month", "month", "year"]).count()

# df_eda_1.to_csv(f"{output_complete_path}/datetime_counts_eda.csv")
df_datetime.write.mode("overwrite").csv(f"{output_complete_path}/datetime_counts_eda.csv", header=True)

# Group by 'author' and 'subreddit' and count the occurrences
author_counts = df.groupBy(['author', 'subreddit']).count()

# Create a window partitioned by 'subreddit' and ordered by the count in descending order
windowSpec = Window.partitionBy('subreddit').orderBy(col('count').desc())

# Add a row number for each row within the partition
author_ranked = author_counts.withColumn("rank", row_number().over(windowSpec))

# Filter to get the top 11 authors for each subreddit
top_authors_each_subreddit = author_ranked.filter(col('rank') <= 11)
top_authors_pd = top_authors_each_subreddit.toPandas()
filtered_authors = top_authors_pd[top_authors_pd["subreddit"].isin(["anime", "television", "movies"])]
filtered_authors = filtered_authors[~filtered_authors["author"].isin(["[deleted]"])]
filtered_authors.to_csv(f"{CSV_DIR}/author_eda.csv")

# Group by 'day_of_month' and 'subreddit', then calculate the average count
day_of_month_avg = df.groupBy(["day_of_month", "subreddit"]).agg(F.avg("score").alias("average_score"))
day_of_month_pd = day_of_month_avg.toPandas()
day_of_month_pd.to_csv(f"{CSV_DIR}/day_of_month_avg_eda.csv")

# Stop Spark session
spark.stop()
