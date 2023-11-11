import argparse
import os
import logging
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

logging.basicConfig(level=logging.INFO)

# Parse Inputs
parser = argparse.ArgumentParser()
parser.add_argument("--platform")
parser.add_argument("--input_object_store_base_url")
parser.add_argument("--input_path")
parser.add_argument("--output_object_store_base_url")
parser.add_argument("--output_path")
parser.add_argument("--subreddits")
args = parser.parse_args()

logging.info(args.platform)
logging.info(args.input_object_store_base_url)
logging.info(args.input_path)
logging.info(args.output_object_store_base_url)
logging.info(args.output_path)
logging.info(args.subreddits)

input_complete_path = f"{args.input_object_store_base_url}{args.input_path}"
output_complete_path = f"{args.output_object_store_base_url}{args.output_path}"

logging.info(input_complete_path)
logging.info(output_complete_path)

spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
logging.info(f"spark version = {spark.version}")

if args.platform == "azureml":
    # Add the Blob SAS credentials
    blob_account_name = "marckvnonprodblob"
    blob_container_name = "bigdata"
    blob_sas_token = "sv=2021-10-04&st=2023-10-04T20%3A02%3A01Z&se=2023-12-31T21%3A02%3A00Z&sr=c&sp=racwdxltf&sig=l%2BbUjYGp1p2IDeyanWtXpDjssBCdW%2B4CJlO4SfPnCEk%3D"
    spark.conf.set(
        f"fs.azure.sas.{blob_container_name}.{blob_account_name}.blob.core.windows.net",
        blob_sas_token,
    )


# Read data from object store
logging.info(f"going to read {input_complete_path}")
df_in = spark.read.parquet(input_complete_path)
df_in_ct = df_in.count()
logging.info(f"finished reading files...")

# filter the dataframe to only keep the subreddits of interest
subreddits = [s.strip() for s in args.subreddits.split(",")]
filtered = df_in.where(col("subreddit").isin(subreddits))
filtered_ct = filtered.count()

# save the filtered dataframes so that these files can now be used for future analysis
logging.info(f"going to write {output_complete_path}")

logging.info(f"Read in {df_in_ct} records, wrote out {filtered_ct} records.")
filtered.write.mode("overwrite").parquet(output_complete_path, compression="zstd")

spark.stop()
