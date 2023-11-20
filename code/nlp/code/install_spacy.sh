#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install spacy
pip install spacy

# Download the required spacy model
python -m spacy download en_core_web_sm
