# Entertainment Subreddit Analysis

This research project analyzes how people discuss movies, television, and anime on Reddit. It examines when communities are active, which posts and authors draw engagement, and how Reddit recommendations compare with ratings from Rotten Tomatoes and MyAnimeList. It also evaluates models for classifying posts and predicting engagement.

[Introduction](website-source/index.qmd) · [Exploratory analysis](website-source/eda.qmd) · [NLP report](website-source/nlp.qmd) · [Machine learning report](website-source/ml.qmd) · [Conclusion](website-source/conclusion.qmd)

## Data and workflow

The project introduction reports source volumes of 412 GB of submissions and 918 GB of comments. The engagement report covers January 2021 through April 2023 for `r/movies`, `r/television`, and `r/anime`. Separate extraction jobs filter recommendation communities including `r/MovieSuggestions`, `r/televisionsuggestions`, and `r/Animesuggest`. The checkout includes selected CSV inputs and analysis exports, interactive plots, and rendered reports. The raw Reddit Parquet corpus and full cloud intermediates are required separately for a rerun.

| Stage | What it does | Entry point |
| --- | --- | --- |
| Extraction | Filters source Parquet by subreddit and writes filtered Parquet to object storage using Azure ML or SageMaker Spark jobs. | [Extraction notebooks and scripts](code/data_extraction/) |
| Cleaning and feature engineering | Profiles missing values, derives calendar features, and prepares subreddit, time, and author summaries for analysis. | [Cleaning notebooks and scripts](code/data_cleaning/) |
| Exploratory analysis | Compares posting activity, scores, comment counts, media presence, post length, and top contributors. | [EDA notebook](code/eda/project_eda.ipynb) |
| Natural language processing | Explores word frequency and TF-IDF, extracts titles from recommendation comments with named-entity recognition and pattern matching, and analyzes review sentiment. The reports compare recommendations with [Rotten Tomatoes](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews) and [MyAnimeList](https://www.kaggle.com/datasets/azathoth42/myanimelist) data. | [NLP notebooks and scripts](code/nlp/) |
| Machine learning | Models movie-versus-anime category, comment-based popularity, and Reddit score. The experiments cover random forests, logistic regression, decision trees, and linear regression. | [ML notebooks and scripts](code/ml/) |
| Reporting | Presents the analysis, figures, executive summaries, and conclusions as a Quarto website. | [Quarto source](website-source/) and [rendered site](docs/index.html) |

## Reported findings

- The EDA found no clear recurring engagement cycle. The report identifies a spike in movie posts in early 2021 and treats a major release as a possible explanation. Shorter posts tended to draw more comments, posts marked as having media tended to receive higher scores and more comments, and score and comment count were positively correlated. These are observational results. [EDA report](website-source/eda.qmd)
- Frequently used words in movie and anime discussions included positive terms such as “like” and “good.” Frequently recommended titles spanned a range of external ratings. The reports describe patterns in the sampled communities. User demographics were unverified. [NLP report](website-source/nlp.qmd) · [Conclusion](website-source/conclusion.qmd)
- The tuned random forest reached **75.9% test accuracy** on movie-versus-anime classification. The decision tree reached **87.2% test accuracy** in the popularity experiment, where popularity was defined using comment counts and the analysis used downsampling. The report’s score-regression table lists test RMSE of **1,385.97** for linear regression and **1,447.39** for the decision tree. These figures are from the checked-in report. [ML report](website-source/ml.qmd)

## Tech stack and prerequisites

| Task | Tools and access |
| --- | --- |
| Read the results | Git to clone the repository and a browser to view the bundled HTML. |
| Work with notebooks | A Jupyter-compatible Python environment. Notebook imports include PySpark, Spark NLP, NLTK, pandas, NumPy, scikit-learn, Matplotlib, Plotly, Seaborn, Altair, Pillow, and FuzzyWuzzy. Install dependencies for the specific notebook you run. |
| Run cloud jobs | For AWS, a SageMaker execution role, accessible S3 inputs and outputs, and the SageMaker SDK and Boto3 where imported. For Azure, an Azure ML workspace, Azure ML and Azure Identity Python packages, and authorized access to Blob Storage. |
| Build the SageMaker NLP image | Docker, the AWS CLI, `jq`, and permission to pull the base image and push to the destination ECR repository. The [Dockerfile](code/nlp/Dockerfile) uses a SageMaker Spark 3.4 CPU image with Python 3.9 and installs Spark NLP 5.1.3 and NLTK. |
| Render the report | Quarto. The project type, navigation, styling, and HTML resources are configured in [`_quarto.yml`](website-source/_quarto.yml). |

## Installation and setup

Clone the repository using its configured GitHub remote.

```bash
git clone https://github.com/Sangeetha-238/BigData_Reddit_Analysis.git
cd BigData_Reddit_Analysis
```

The repository has no shared dependency manifest, lockfile, or package installation command. Use the imports and setup cells in your chosen notebook to identify its dependencies. The container pins Spark NLP for its own environment. A single compatible set of versions for all notebooks has not been verified.

## Configuration for a cloud rerun

- **AWS.** Review the [SageMaker extraction notebook](code/data_extraction/sagemaker_data_extraction.ipynb) for processing job arguments, input locations, and output prefixes. Set your own S3 paths, execution role, region, and processing image where the notebooks require them. Many scripts also embed S3 paths directly. Update downstream readers to use the same paths as their upstream writers. Several jobs use overwrite mode, so configure separate output prefixes for reruns.
- **Azure.** Start with the [submissions](code/data_extraction/azureml-job-submissions.ipynb), [comments](code/data_extraction/azureml-job-comments.ipynb), and [cleaning](code/data_cleaning/azureml-job-data_cleaning.ipynb) job definitions. Set your workspace and storage locations, and replace the expired hard-coded Blob SAS with authorized credentials before using the Azure extraction script.
- **External inputs and artifacts.** Obtain the Reddit Parquet inputs and any external source files required by the selected notebook. Check its read paths against the files in [`data/csv/`](data/csv/) and point local reads to your copies. Verify access to the pretrained NLP models used by the relevant jobs.
- **Optional container.** [`build_and_push.sh`](code/nlp/build_and_push.sh) rewrites the Dockerfile in its working directory and pushes an image to ECR. Review the base image, region, repository, and tag before running it. The [separate spaCy helper](code/nlp/code/install_spacy.sh) installs spaCy and downloads `en_core_web_sm`.
- **Reproduction gaps.** The [second ML notebook](code/ml/project_ml2.ipynb) references `ml_train_RFR.py`, which is absent. The report and checked-in notebooks and scripts also differ in how the popularity and score experiments are mapped to files. Check each job’s estimator, target column, and input/output paths before relying on the report’s notebook links for a rerun. Access to the original cloud resources and a complete end-to-end run are unverified.

## Usage

1. **Read the existing results.** Open [`docs/index.html`](docs/index.html) from the checkout in a browser, keeping the `docs/` directory intact so relative assets can load. The report source is also available through the links at the top of this README.
2. **Inspect the analysis.** Open an EDA, NLP, or ML notebook in a Jupyter-compatible editor. For cells that consume bundled CSVs, set the input paths to the files under `data/csv/`. Other cells depend on cloud artifacts or submit jobs.
3. **Rerun selected stages.** Configure the cloud jobs first, then run extraction and cleaning before the downstream notebooks that depend on their outputs. Within an ML experiment, prepare features and split the data before training and evaluating the model. Check the individual notebook before executing cloud job submission cells.

To render the report source locally from the repository root, use the Quarto project directory below. The bundled HTML can be read directly. A fresh render and completeness of its embedded assets have not been verified.

```bash
quarto render website-source
```

## Repository structure

| Path | Contents |
| --- | --- |
| `code/data_extraction/` | Azure ML and SageMaker extraction notebooks and PySpark job scripts |
| `code/data_cleaning/` | Data preparation, time and author summaries, and an Azure ML cleaning job |
| `code/eda/` | Exploratory analysis notebook and its assets |
| `code/nlp/` | Text, sentiment, and recommendation notebooks, Spark job code, and container tooling |
| `code/ml/` | Feature engineering, training, and evaluation notebooks and scripts |
| `code/html_rendered/` | HTML exports of analysis notebooks |
| `data/csv/`, `data/plots/` | Checked-in CSV inputs and exports, plus interactive HTML figures |
| `website-source/` | Quarto sources, configuration, report assets, and a generated `_site/` directory |
| `docs/` | Rendered report and its assets |

## Testing and validation

The checkout contains no automated test suite or CI workflow. The ML analysis documents holdout evaluation. The [feedback report](website-source/discussion.qmd) records that cross-validation and repeated downsampling were not performed.

For analysis changes, rerun the affected stages with your configured inputs and compare schemas, counts, and metrics with the relevant report or exported artifact. For report changes, render the site and check navigation and embedded figures in a browser.

## Contributing

Open an issue or pull request with the affected notebook, script, or report and describe the inputs, environment, and steps used to verify the change. Keep credentials and private storage data out of contributions.

## Authors and license

Ramdayal Nirmal Rewaria (rr1220), Samiksha Rajendra Thikekar (st1140), and Sangeetha Kanithahalli Ramesh (sk2224).

The repository includes an [MIT license](LICENSE). The [Quarto site configuration](website-source/_quarto.yml) declares CC BY-NC 4.0 for website content.
