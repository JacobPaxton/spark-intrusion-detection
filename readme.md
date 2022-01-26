# Spark Intrusion Detection
This project uses Spark to analyze a network intrusion dataset. I specifically want to use PySpark with anomaly detection and clustering techniques to successfully classify network traffic as normal or anomalous.

# Data Background
This dataset is TCP/IP dump data for a simulated US Air Force LAN and details various aspects of each connection, including duration, protocol, and more. There are 4 million rows and 41 features with a separate column indicating either a normal connection or one of the various network attacks being performed in the observation.

Here is the official description of this dataset: https://kdd.ics.uci.edu/databases/kddcup99/task.html

This dataset is available for download here: https://www.kaggle.com/galaxyh/kdd-cup-1999-data

# Objectives
1. Use Spark to accomplish wrangling and queries
2. Use Spark's MLlib to split data
3. Export columns for statistical testing and visualization as required
4. Use Spark to engineer features using anomaly detection techniques
5. Use Spark's MLlib to create features using outlier-detection clustering
6. Use Spark's MLlib to create classification models using engineered features
7. Use Spark's MLlib to evaluate models, tune hyperparameters, and evaluate on sequestered split
8. Output Spark aggregation to CSVs for Tableau visualization
9. Present work and findings in a Jupyter Notebook

# Plan
## Minimum Viable Product (MVP)
- [x] Set up a Spark virtual environment
- [x] Ingest the full intrusion detection dataset
- [x] Gain initial awareness using basic distributions and value counts
- [x] Determine the path forward and next steps
- [x] Reduce 'smurf' and 'neptune' attack classes by 95% through isolating their rows and sampling
    * Use sampleBy with python dict containing each class's fraction
- [x] Convert dataset to binary classes by combining all attack classes into category 'anomalous'
- [x] Split the reduced dataset 50%-30%-20% for model training, validation, and testing
    * Stratify the target column between train and validate split
- [x] Create initial hypotheses using domain knowledge, conduct statistical testing to answer them
    * Categorical target: Use Chi Square tests and Comparison of Means tests
    * Hypotheses: Attack categories have certain expected appearance in the dataset
    * Export results to Pandas for quick visualization in Seaborn
- [x] Use initial-hypotheses features that proved significant as model features
- [x] Build and fit a classification model on selected features in train split
- [x] Evaluate model performance on in-sample (train) and out-of-sample (validate) data
- [x] Push work to scripts
- [x] Report all work in MVP final notebook
## Model Deployment MVP
- [x] Push model to a Docker container
- [x] Pass inputs to container
## Post-MVP Iteration
- [ ] Conduct statistical testing on categorical features to identify potential candidates for modeling
- [ ] Check distributions on continuous features using target as hue to identify features for clustering
- [ ] Use feature engineering to create more-predictive features
- [ ] Use Tableau to visualize engineered features
- [ ] Build and fit more classification models using new features
- [ ] Use Grid Search to optimize hyperparameters
- [ ] Evaluate tuned models on train and validate
- [ ] Select best model and evaluate model on test
- [ ] Report all work in post-MVP final notebook
## Post-MVP Model Deployment
- [ ] Push model to an AWS partition through Docker
- [ ] Pass sequestered unlabeled dataset to model in batch
- [ ] Report results in production notebook