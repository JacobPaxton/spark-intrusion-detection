# Spark Intrusion Detection
This project uses Spark to analyze a network intrusion dataset. I specifically want to use PySpark with anomaly detection and clustering techniques to successfully classify network traffic as normal or anomalous.

# Data Background
This dataset is TCP/IP dump data for a simulated US Air Force LAN and details various aspects of each connection, including duration, protocol, and more. There are 4 million rows and 41 features with a separate column indicating either a normal connection or one of the various network attacks being performed in the observation.

Here is the official description of this dataset: https://kdd.ics.uci.edu/databases/kddcup99/task.html

This dataset is available for download here: https://www.kaggle.com/galaxyh/kdd-cup-1999-data

# Objectives
1. Use Spark for data wrangling, queries, aggregation, split, and sampling
    * Use all normal rows, drop 95% of DOS attack rows, and use all non-DOS attack traffic: roughly 1.1m rows
    * Split the roughly-1.1m rows into 50%-30%-20% (train-validate-test) splits, stratify train and validate split
2. Use anomaly detection techniques to build features through risk thresholds
    * Domain knowledge: Manual thresholding
    * Exploration: Analyze each attack's metrics against normal data, document deviations, create features
3. Use clustering techniques to capture any unusual connections not directly addressed in anomaly detection
    * Intended clustering technique: DBSCAN, for its inbuilt outlier classification
4. Deliver insights comparing attacks to normal data in different metrics
    * Cover the 4 major attack categories and their appearance in the dataset
5. Build classification models using selected and engineered features
    * Fit on train split
    * Prioritized algorithm: SVM, but use other algorithms as well
6. Evaluate models, tune hyperparameters, and select best-performing model
    * Evaluate on train (in-sample) and validate (out-of-sample) data splits
7. Evaluate best-performing model on sequestered test split

# Additional Objective
In addition to this analysis and modeling, I would like to take some time to deploy the model onto an Azure partition and process the additional unlabeled dataset in a batch, simulating a detected breach and using the model to identify the offending traffic.

# Plan
1. 