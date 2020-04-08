from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import preprocessing as dp
import constants
import datetime
import matplotlib.pyplot as plt

def predictor(model, grid, training, testing, model_name, label_col):
    # 4.3 evaluator
    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol=label_col)

    # 4.4 cross validation
    cv = CrossValidator(estimator=model, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3, parallelism=8,
                        seed=0)
    model = cv.fit(training)

    # 4.5 prediction
    prediction = model.transform(testing).cache()
    # prediction.show(10)

    # 5.1 print best hyper-parameter grid
    accuracy = evaluator.evaluate(prediction, {evaluator.metricName: "accuracy"})
    print(model_name + " accuracy: %.8f" % accuracy)
    f1_score = evaluator.evaluate(prediction, {evaluator.metricName: "f1"})
    print(model_name + " f1: %.8f" % f1_score)

    # return best model to print best estimated hyper-parameters
    return model.bestModel


def nb_classify(training, testing, labelCol, featuresCol):
    # 5.1 train the NB model
    nb_model = NaiveBayes(labelCol=labelCol, featuresCol=featuresCol)

    # 5.2 nb parameter grid
    nb_grid = ParamGridBuilder().addGrid(nb_model.smoothing, [0.0, 0.4, 0.8, 1.0]).build()

    # 5.3 predict
    bestModel = predictor(nb_model, nb_grid, training, testing, 'Naive Bayes Classification', labelCol)

    # 5.4 print best hyper-parameter
    print("Best hyper-parameter: ")
    print("Smoothing: " + str(bestModel._java_obj.getSmoothing()))


def rf_classify(training, testing, labelCol, featuresCol):
    # 4.1 train the RF model
    rf_model = RandomForestClassifier(labelCol=labelCol, featuresCol=featuresCol,seed=0)

    # 4.2 rf parameter grid
    # rf_grid = ParamGridBuilder().addGrid(rf_model.maxDepth, [3, 5, 10])\
    #     .addGrid(rf_model.numTrees, [5, 10, 20])\
    #     .addGrid(rf_model.maxBins, [15, 20, 25]).build()

    rf_grid = ParamGridBuilder().addGrid(rf_model.maxDepth, [3, 5, 10]) \
        .addGrid(rf_model.numTrees, [10, 50, 100]).build()

    # 4.3 predict
    bestModel = predictor(rf_model, rf_grid, training, testing, 'Random Forest Classification', labelCol)

    # 4.4 print best hyper-parameter
    print("Best hyper-parameter: ")
    print("maxDepth: " + str(bestModel._java_obj.getMaxDepth()))
    print("NumTrees: " + str(bestModel._java_obj.getNumTrees()))
    # print("maxBins: " + bestModel._java_obj.getMaxBins())
    return bestModel


def knn_classify(training, testing, labelCol, featuresCol):
    # convert dataframe to numpy
    training = training.toPandas()
    testing = testing.toPandas()
    if featuresCol.startswith("pca"):
        # convert dense matrix into multi-dimention array
        x_train_series = training[featuresCol].values.reshape(-1, 1)
        x_train = np.apply_along_axis(lambda x: x[0], 1, x_train_series)
        x_test_series = testing[featuresCol].values.reshape(-1, 1)
        x_test = np.apply_along_axis(lambda x: x[0], 1, x_test_series)
    else:
        # convert sparse matrix into multi-dimention array
        x_train_series = training[featuresCol].apply(lambda x: np.array(x.toArray())).values.reshape(-1, 1)
        x_train = np.apply_along_axis(lambda x: x[0], 1, x_train_series)
        x_test_series = testing[featuresCol].apply(lambda x: np.array(x.toArray())).values.reshape(-1, 1)
        x_test = np.apply_along_axis(lambda x: x[0], 1, x_test_series)
    y_train = training[labelCol].values
    y_test = testing[labelCol].values

    knn_classifier = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 10, 25]}

    # uses grid search for cross-validation
    gscv = GridSearchCV(estimator=knn_classifier,
                        param_grid=param_grid,
                        cv=3,
                        n_jobs=8)
    # training models
    gscv.fit(x_train, y_train)

    # get prediction
    y_pred = gscv.predict(x_test)

    # evaluation
    print("KNeighborsClassifier accuracy: %.8f" % (accuracy_score(y_test, y_pred)))
    print("KNeighborsClassifier f1      : %.8f" % (f1_score(y_test, y_pred)))

    # print best estimator params
    print("Best estimator :: ", gscv.best_estimator_)


def plot_importance(importance, feature_names):
    plt.figure(figsize=(16, 16))
    plt.title("Feature importances")
    plt.barh(range(len(importance)), importance, color="r", align="center")
    plt.yticks(range(len(importance)), feature_names)
    plt.ylim([-1, len(importance)])
    plt.savefig(constants.OUTPUT_DIR + "/feature_imp")


if __name__ == '__main__':
    spark = dp.init_spark()
    original_df = dp.load_data_from_file(spark, constants.MUSHROOM_DATASET_withLABEL)
    df = dp.drop_features(original_df)
    df, col_names = dp.deal_missing_values(df)
    df, string_indexer_df = dp.encoding_data(df)
    (training, testing) = df.randomSplit([0.8, 0.2], seed=0)
    training, testing = dp.apply_pca(training, testing)

    print("------ result of original features ------")
    start_time = datetime.datetime.now()
    nb_classify(training, testing, training.schema.names[0], training.schema.names[1])
    rf_classify(training, testing, training.schema.names[0], training.schema.names[1])

    knn_classify(training, testing, training.schema.names[0], training.schema.names[1])
    end_time = datetime.datetime.now()
    time_take = int((end_time - start_time).total_seconds())
    print("time taken: ", time_take,  " seconds")
    print()
    # training PCA features
    print("------ result of PCA features ------")
    scaler = MinMaxScaler(inputCol=training.schema.names[2], outputCol="scaledPCAFeatures")
    scalerModel = scaler.fit(training)
    training = scalerModel.transform(training)
    testing = scalerModel.transform(testing)

    # naive bayes cant deal with negative input features we skip PCA features here
    start_time = datetime.datetime.now()
    nb_classify(training, testing, training.schema.names[0], "scaledPCAFeatures")
    rf_classify(training, testing, training.schema.names[0], training.schema.names[2])
    knn_classify(training, testing, training.schema.names[0], training.schema.names[2])
    end_time = datetime.datetime.now()
    time_take = int((end_time - start_time).total_seconds())
    print("time taken: ", time_take, " seconds")

    # research on feature importance
    print("-------------------feature importance------------------------")
    (training_indexer, testing_indexer) = string_indexer_df.randomSplit([0.8, 0.2], seed=0)
    best_rf = rf_classify(training_indexer, testing_indexer, "class_index", "feature_vec")
    importance = best_rf.featureImportances
    plot_importance(importance, col_names[1:])





