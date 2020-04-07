import copy

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, OneHotEncoderEstimator, StringIndexer, PCA
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from classification import nb_classify, rf_classify, knn_classify
from src import constants


def undersample_majority():
    '''
    this function is for undersampling the majority if data imbalance exists
    :return:
    '''
    pass


def plot_correlation_matrix(pearsonCorr, threshold=0.7):
    plt.figure(figsize=(8, 8))
    strong_index = [item for item in np.argwhere(pearsonCorr >= threshold) if item[0] != item[1] and item[0] < item[1]]
    plt.imshow(pearsonCorr, interpolation='nearest', cmap=plt.cm.bwr)
    plt.colorbar()
    plt.savefig("." + constants.OUTPUT_DIR + "/correlation_matrix")
    return strong_index


spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
# read from file
df = spark.read.csv("." + constants.MUSHROOM_DATASET_withLABEL, header=True).toPandas()
# step 1: convert labels from str into integer
df["class"] = df["class"].replace(["e", "p"], [0, 1]).astype(int)

# step 2: drop features columns with only one distinct value
df = df[[column for column in list(df) if len(df[column].unique()) > 1]]

# step 3: impute the missing value by most frequent item in that column
cols = df.columns
si = SimpleImputer(missing_values="?", strategy="most_frequent")
trans_df = si.fit_transform(df)
df = spark.createDataFrame(pd.DataFrame(data=trans_df, columns=cols)).cache()
ins_total = df.count()

# step 4: Filter into 2 sets based on label to deal with data imbalance (not implemented yet)
rdd = spark.read.csv("." + constants.MUSHROOM_DATASET_withLABEL, header=True).rdd.persist()
number_e = rdd.filter(lambda x: x[0] == 'e').count()
number_p = rdd.filter(lambda x: x[0] == 'p').count()
lable_ration = max(number_e, number_p) / (number_e + number_p)
if lable_ration > .7:
    undersample_majority()

# step 5: one-hot encoding for all features
feature_cols = df.schema.names[1:]
label = df.schema.names[0]
feature_trans = [item + "_index" for item in feature_cols]
# 5.1 string indexer
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index").fit(df) for col in feature_cols]
cols_for_pearson = [label] + feature_trans
# 5.2 vector assembler
vecAssembler = VectorAssembler(inputCols=cols_for_pearson, outputCol="features")
# 5.3 one-hot encoding
cols_name_vec = [item + "_vec" for item in feature_trans]
encoder = OneHotEncoderEstimator(inputCols=feature_trans, outputCols=cols_name_vec)
# 5.3 pipeline
pipeline = Pipeline(stages=indexers + [vecAssembler, encoder])
df_r = pipeline.fit(df).transform(df).cache()
# 5.4 get correlation matrix then plot it and get correlation paris (index 0: label; index 1-end: features)
pearsonCorr = Correlation.corr(df_r, column="features", method='pearson').collect()[0][0].values.reshape(
    len(feature_trans) + 1, len(feature_trans) + 1)
# *here we use 0.7 as the strong correlation threshold
strong_corr_index = plot_correlation_matrix(pearsonCorr, threshold=0.7)

df_trans_list = []

for corr_pair in strong_corr_index:
    dc_cols_name_vec = copy.copy(cols_name_vec)
    # this correlated pair is associated with label
    # in such cases, just remove the correlated feature from features list
    if corr_pair[0] == 0:
        dc_cols_name_vec.pop(corr_pair[1] - 1)
        vecAssembler = VectorAssembler(inputCols=[label] + dc_cols_name_vec, outputCol="features_training")
        df_trans_list.append(vecAssembler.transform(df_r).select("class", "features_training"))
    # this correlated pair is between features
    # in such cases, remove either of both and append to the list
    else:
        dc_cols_name_vec.pop(corr_pair[0] - 1)
        vecAssembler = VectorAssembler(inputCols=[label] + dc_cols_name_vec, outputCol="features_training")
        df_trans_list.append(vecAssembler.transform(df_r).select("class", "features_training"))
        dc_cols_name_vec = copy.copy(cols_name_vec)
        dc_cols_name_vec.pop(corr_pair[1] - 1)
        vecAssembler = VectorAssembler(inputCols=[label] + dc_cols_name_vec, outputCol="features_training")
        df_trans_list.append(vecAssembler.transform(df_r).select("class", "features_training"))

pca = PCA(k=10, inputCol="features_training", outputCol="pca_features")
for df in df_trans_list:
    # step 6: split training and testing set
    (training, testing) = df.randomSplit([0.8, 0.2], seed=0)
    # step 6: Project features into a lower dimensional space by using PCA
    pca_model = pca.fit(training)
    training = pca_model.transform(training).cache()
    # transform testing model by using the PCA transformed training model
    testing  = pca_model.transform(testing).cache()
    print("training model: " + str(df_trans_list.index(df)))
    print()
    # training original features
    print("------ result of original features ------")
    nb_classify(training, testing, training.schema.names[0], training.schema.names[1])
    rf_classify(training, testing, training.schema.names[0], training.schema.names[1])
    knn_classify(training, testing, training.schema.names[0], training.schema.names[1])
    print()
    # training PCA features
    print("------ result of PCA features ------")
    # naive bayes cant deal with negative input features we skip PCA features here
    # nb_classify(training, testing, df.schema.names[0],df.schema.names[2])
    rf_classify(training, testing, training.schema.names[0], training.schema.names[2])
    knn_classify(training, testing, training.schema.names[0], training.schema.names[2])
    print()

# # 4.1 train the RF model
# rf_model = RandomForestClassifier(labelCol='label_index', featuresCol='features')
#
# # 4.2 rf parameter grid
# # rf_grid = ParamGridBuilder().addGrid(rf_model.maxDepth, [4, 5, 6])\
# #     .addGrid(rf_model.numTrees, [50, 75, 100])\
# #     .addGrid(rf_model.maxBins, [15, 20, 25]).build()
#
# rf_grid = ParamGridBuilder().addGrid(rf_model.maxDepth, [4, 5, 6])\
#     .addGrid(rf_model.numTrees, [100]).build()
#
# # 4.3 predict
# predictor(rf_model, rf_grid, training, testing, 'Random Forest Classification')
#
# # 5.1 train the NB model
# nb_model = NaiveBayes(labelCol='label_index', featuresCol='features')
#
# # 5.2 nb parameter grid
# nb_grid = ParamGridBuilder().addGrid(nb_model.smoothing, [0.0, 0.4, 0.8, 1.0]).build()
#
# # 5.3 predict
# predictor(nb_model, nb_grid, training, testing, 'Naive Bayes Classification')
#
#
# pipeline = Pipeline(stages=indexers + [encoder])
# df_r = pipeline.fit(df).transform(df)
#
# print("--------------------------4----------------------------------")
# vecAssembler = VectorAssembler(inputCols=["class"] + feature_trans, outputCol="features")
# features = vecAssembler.transform(df_r)
# pearsonCorr = Correlation.corr(features,column= "features", method='pearson').collect()[0][0]
# print(pearsonCorr.size)
#
# # for i in cols[1:]:
#
#     # missing_num = df.filter(df[i] == "?").count()
#     # if missing_num / ins_total > .1:
#     #     df = df.drop(i)
#     #     print("Drop [%s] column since the missing values in this column are more than threshold." % i)
#
