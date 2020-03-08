import constants

import numpy as np
from pyspark.sql import SparkSession
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pathlib import Path

import sklearn


def undersample_majority():
    '''
    this function is for undersampling the majority if data imbalance exists
    :return:
    '''
    pass


class Preprocessing:
    data = []
    targets = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []


    def __init__(self):
        spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        df = spark.read.csv("."+constants.MUSHROOM_DATASET_withLABEL,header = True).persist()
        ins_total = df.count()
        rdd = spark.read.csv("."+constants.MUSHROOM_DATASET_withLABEL,header = True).rdd.persist()
        # Filter into 2 sets based on label to deal with data imbalance
        number_e = rdd.filter(lambda x : x[0] == 'e').count()
        number_p = rdd.filter(lambda x : x[0] == 'p').count()
        lable_ration = max(number_e,number_p)/(number_e+number_p)
        if lable_ration > .7:
            undersample_majority()

        # deal with missing values on features
        print(df.columns)
        for i in df.columns:
            missing_num = df.filter(df[i] == "?").count()
            if missing_num/ins_total > .1:
                df = df.drop(i)
                print("Drop [%s] column since the missing values in this column are more than threshold." % i)







        ### detect if there is data imbalance
        ### if we find one single class that takes more than 70%
        ### we will undersample the majority.
        # self.targets = self.data[:, 0]
        # self.targets


if __name__ == '__main__':
    pre = Preprocessing()
    # print(pre.data)
