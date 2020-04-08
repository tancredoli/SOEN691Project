import copy

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, OneHotEncoderEstimator, StringIndexer, PCA
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants


def plot_correlation_matrix(pearsonCorr, column_names):
    plt.figure(figsize=(8, 8))
    plt.imshow(pearsonCorr, interpolation='nearest', cmap=plt.cm.bwr)

    plt.xticks(np.arange(0, 21, 1).tolist(), column_names, rotation='vertical')
    plt.yticks(np.arange(0, 21, 1).tolist(), column_names)
    plt.colorbar()
    plt.savefig(constants.OUTPUT_DIR + "/correlation_matrix_new")
    return


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def load_data_from_file(spark, filename):
    df = spark.read.csv(filename, header=True)
    column_names = df.schema.names
    return df, column_names


def drop_features(df):
    column_names = df.schema.names
    for name in column_names:
        if df.select(name).distinct().count() < 2:
            df = df.drop(name)
    return df


def deal_missing_values(df):
    total_count = df.count()
    column_names = df.schema.names
    for name in column_names:
        if (df.filter(df[name] == "?").count()) / total_count > 0.2:
            df = df.drop(name)
        else:
            df = df.filter(df[name] != "?")
    return df


def encoding_data(df):
    column_names = df.schema.names
    column_indexes = [item + "_index" for item in column_names]
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_index").fit(df) for col in column_names]

    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df).cache()

    df = VectorAssembler(inputCols=column_indexes, outputCol="corr_vec").transform(df).cache()

    strong_index = get_correlation_matrix(df, column_names, 0.6)
    dc_column_indexes = copy.copy(column_indexes)
    for corr_pair in strong_index:
        if corr_pair[0] != 0:
            # this correlated pair is between features
            # in such cases, remove either of both and append to the list
            dc_column_indexes.pop(corr_pair[0])
    column_vecs = [item + "_vec" for item in dc_column_indexes]
    df = OneHotEncoderEstimator(inputCols=dc_column_indexes, outputCols=column_vecs).fit(df).transform(df).cache()
    df = VectorAssembler(inputCols=column_vecs, outputCol="features").transform(df).cache()
    return df.select("class_index", "features")


def apply_pca(training, testing):
    pca = PCA(k=3, inputCol="features", outputCol="pca_features").fit(training)
    training = pca.transform(training).cache()
    testing = pca.transform(testing).cache()
    return training, testing


def get_correlation_matrix(df, column_names, threshold):
    pearsonCorr = Correlation.corr(df, column="corr_vec", method='pearson').collect()[0][0].values.reshape(
        len(column_names), len(column_names))
    plot_correlation_matrix(pearsonCorr, column_names)
    strong_index = [item for item in np.argwhere(pearsonCorr >= threshold) if item[0] != item[1] and item[0] < item[1]]
    return strong_index

