# Use a base image compatible with your Spark and Python versions
FROM python:3.8-slim

# Install Java (required for Spark)
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;

# Install spaCy
RUN pip install spacy

# Download the spaCy English model
RUN python -m spacy download en_core_web_sm
