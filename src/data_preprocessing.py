import constants

import numpy as np
from pyspark.sql import SparkSession
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pathlib import Path

import sklearn
import pandas


def preprocess_data():
    load_data()


def load_data():
    data = pandas.read_csv(constants.MUSHROOM_DATASET_withLABEL)
    data_label = data["class"]
    data_features = data.drop(["class"], axis=1)
    data_label = data_label.applymap(lambda x: x == "p")

    print(data_label.head(5))


if __name__ == '__main__':
    preprocess_data()
