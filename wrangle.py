import pandas as pd
import pyspark
from pyspark.sql.functions import *

def prep_explore():
    '''
        Acquire network intrusion dataset,
        Reduce DOS attack class rows by 95%,
        Group all attack classes into 'anomalous' class,
        Split data 50%-30%-20% for train, validate, and test splits,
        Return original df and the three splits.
    '''
    # set up environment, ingest data
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = spark.read.csv('kddcup.data.corrected', header=True)

    # set sample size to 5% for 'neptune' and 'smurf' attack classes
    fraction_df = (
            df.select('target')
            .distinct()
            .withColumn('fraction', 
                when((df.target == 'neptune.') | (df.target == 'smurf.'), 0.05)
                .otherwise(1)
            )
    )

    # convert fractions df to dict, use dict in sampleBy
    df = df.sampleBy('target', fraction_df.toPandas().set_index('target').to_dict()['fraction'])

    # convert target column to 'normal' and 'anomalous' classes
    df = df.withColumn('target', when(df.target != 'normal.', 'anomalous').otherwise('normal'))

    # split data 50%-30%-20%
    train, validate, test = df.randomSplit([0.5, 0.3, 0.2], seed=42)

    # return original and splits
    return df, train, validate, test