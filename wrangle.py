import pandas as pd
import pyspark
from pyspark.sql.functions import *
from pyspark.sql.types import *

def prep_model_MVP():
    '''
        Acquire network intrusion dataset,
        Limit features to exploration-selected 'srv_count' and 'num_failed_logins',
        Fix column dtypes,
        Reduce DOS attack class rows by 95%,
        Group all attack classes into 'anomalous' class,
        Split data 50%-30%-20% for df, validate, and test splits,
        Return the three splits.
    '''
    # set up environment, ingest data
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = spark.read.csv('kddcup.data.corrected', header=True)

    # select the two MVP features and the target
    df = df.select('srv_count', 'num_failed_logins', 'target')

    # fix dtypes
    df = df.withColumn('num_failed_logins', df.num_failed_logins.cast(IntegerType()))\
           .withColumn('srv_count', df.srv_count.cast(IntegerType()))

    # convert target to binary classification, limit DOS attack rows 95%
    df = convert_to_binary_class(df)

    # encode target column for 1 = anomaly
    df = df.withColumn('target', when(df.target == 'anomalous', 1).otherwise(0))

    # split data 50%-30%-20%
    train, validate, test = df.randomSplit([0.5, 0.3, 0.2], seed=42)

    # convert the splits to pandas dataframes
    train = train.toPandas()
    validate = validate.toPandas()
    test = test.toPandas()

    # return original and splits
    return train, validate, test


def prep_explore():
    '''
        Acquire network intrusion dataset,
        Fix column dtypes,
        Reduce DOS attack class rows by 95%,
        Group all attack classes into 'anomalous' class,
        Split data 50%-30%-20% for df, validate, and test splits,
        Return original df and the three splits.
    '''
    # set up environment, ingest data
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = spark.read.csv('kddcup.data.corrected', header=True)

    # fix column dtypes
    df = fix_dtypes(df)

    # convert target to binary classification, limit DOS attack rows 95%
    df = convert_to_binary_class(df)

    # split data 50%-30%-20%
    train, _, _ = df.randomSplit([0.5, 0.3, 0.2], seed=42)

    # return original and splits
    return df, train


def convert_to_binary_class(df):
    '''
        Take the network intrustion dataframe, 
        Reduce DOS attack rows by 95%, 
        Convert target to 'normal' and 'anomalous' classes for binary classification,
        Return dataframe.
    '''
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
    df = df.sampleBy('target', fraction_df.toPandas().set_index('target').to_dict()['fraction'], seed=42)

    # convert target column to 'normal' and 'anomalous' classes
    df = df.withColumn('target', when(df.target != 'normal.', 'anomalous').otherwise('normal'))

    return df


def fix_dtypes(df):
    """ Cast every column in the network intrusion dataset as the proper dtype """
    return (df.withColumn('duration', df.duration.cast(IntegerType()))
        .withColumn('protocol_type', df.protocol_type.cast(StringType()))
        .withColumn('service', df.service.cast(StringType()))
        .withColumn('flag', df.flag.cast(StringType()))
        .withColumn('src_bytes', df.src_bytes.cast(IntegerType()))
        .withColumn('dst_bytes', df.dst_bytes.cast(IntegerType()))
        .withColumn('land', df.land.cast(StringType()))
        .withColumn('wrong_fragment', df.wrong_fragment.cast(IntegerType()))
        .withColumn('urgent', df.urgent.cast(IntegerType()))
        .withColumn('hot', df.hot.cast(IntegerType()))
        .withColumn('num_failed_logins', df.num_failed_logins.cast(IntegerType()))
        .withColumn('logged_in', df.logged_in.cast(StringType()))
        .withColumn('num_compromised', df.num_compromised.cast(IntegerType()))
        .withColumn('root_shell', df.root_shell.cast(IntegerType()))
        .withColumn('su_attempted', df.su_attempted.cast(IntegerType()))
        .withColumn('num_root', df.num_root.cast(IntegerType()))
        .withColumn('num_file_creations', df.num_file_creations.cast(IntegerType()))
        .withColumn('num_shells', df.num_shells.cast(IntegerType()))
        .withColumn('num_access_files', df.num_access_files.cast(IntegerType()))
        .withColumn('num_outbound_cmds', df.num_outbound_cmds.cast(IntegerType()))
        .withColumn('is_host_login', df.is_host_login.cast(StringType()))
        .withColumn('is_guest_login', df.is_guest_login.cast(StringType()))
        .withColumn('srv_count', df.srv_count.cast(IntegerType()))
        .withColumn('serror_rate', df.serror_rate.cast(DoubleType()))
        .withColumn('srv_serror_rate', df.srv_serror_rate.cast(DoubleType()))
        .withColumn('rerror_rate', df.rerror_rate.cast(DoubleType()))
        .withColumn('srv_rerror_rate', df.srv_rerror_rate.cast(DoubleType()))
        .withColumn('same_srv_rate', df.same_srv_rate.cast(DoubleType()))
        .withColumn('diff_srv_rate', df.diff_srv_rate.cast(DoubleType()))
        .withColumn('srv_diff_host_rate', df.srv_diff_host_rate.cast(DoubleType()))
        .withColumn('dst_host_count', df.dst_host_count.cast(IntegerType()))
        .withColumn('dst_host_srv_count', df.dst_host_srv_count.cast(IntegerType()))
        .withColumn('dst_host_same_srv_rate', df.dst_host_same_srv_rate.cast(DoubleType()))
        .withColumn('dst_host_diff_srv_rate', df.dst_host_diff_srv_rate.cast(DoubleType()))
        .withColumn('dst_host_same_src_port_rate', df.dst_host_same_src_port_rate.cast(DoubleType()))
        .withColumn('dst_host_srv_diff_host_rate', df.dst_host_srv_diff_host_rate.cast(DoubleType()))
        .withColumn('dst_host_serror_rate', df.dst_host_serror_rate.cast(DoubleType()))
        .withColumn('dst_host_srv_serror_rate', df.dst_host_srv_serror_rate.cast(DoubleType()))
        .withColumn('dst_host_rerror_rate', df.dst_host_rerror_rate.cast(DoubleType()))
        .withColumn('dst_host_srv_rerror_rate', df.dst_host_srv_rerror_rate.cast(DoubleType()))
        .withColumn('target', df.target.cast(StringType()))
    )