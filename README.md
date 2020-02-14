# SOEN691Project
## Abstract
This report will perform analysis on a dataset which contains physical characters of poisonous or edible mushrooms. The dataset is Mushroom Data Set from UCI machine learning repository provided by Jeff Schlimmer. First, we will preprocess this dataset to get the training set and testing set. Then use the training set to train a random forest classifier and a k-means clustering classifier in Spark and Scikit Learn, and try to predict whether a mushroom is safe or poisonous by using a customized cost matrix. Second, we will analyze the results and also compare the accuracy, F1 score and cost of time of these two techniques. 
## Introduction
This dataset was donated to UCI ML dataset in 1987, it has information of 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom. This data set has 22 features and four labels(edible, definitely poisonous, unknown edibility and not recommended), it has 8124 instances and there are missing values in some of those instances. The objective of this project is using KNN and random forest to predict if a kind of mushroom is safe to eat. Because this is a classification problem and both KNN and random forest algorithms perform well for classification tasks. Besides, both algorithms are supported by Spark and can be run parallelly. Here are some problems we need to solve: How to select useful features? How to balance the data? How to deal with missing values? How to represent categories in vectors since they are all characters now? And how to do the normalization? How to reduce the number of labels to two which are edible and poisonous. After this, we need to consider how to apply the methods provided by Spark and Sci-learn on the data. Then, after we get the model, we need to use the model on the test set and analyze the result. The ways of analyzing the result are also something we need to consider. （Related work 这部分有点强行解释，不太清楚related work指的是啥）There is some related work we need to do. For the feature selection, we need to do some research on the relationship between mushroom’s physical characters and it’s poisonous to help us decide the usefulness of a feature. For the result analyzing part, we also need to do some research about how to reasonably evaluate our result.
## Materials and Methods
We will use the Mushroom data set from UCI to train two different classifiers based on KNN and Random Forest algorithms. We are going to divide the whole process into three individual steps: Data Preprocessing, Training and Evaluation.  
### Data Preprocessing

As soon as we get the raw data. The very first one is preprocessing the dataset. There are six step-by-step methods in this part: Data Selection, Missing Values Imputation, Categorical Features Encoding, Data Normalization, Training and Test set separation and Data Balance.

#### Data Selection

There are 22 different physical features for this task while the information extracted from them varies a lot. I.e. Some of the features tell us more useful hints compared with the rest. We are going to select the most valuable features based on different standpoint and evaluate the impact of the different selections to the final results. There are four different labels in the raw dataset which are definitely edible, definitely poisonous, or of unknown edibility and not recommended. Since we believe that the last two of them would disturb the classifier. We are going to discard the raws with these two labels at the beginning. 
 
#### Missing Values Imputation

As we know, real-world datasets often contain missing values which are not usually acceptable for most machine learning algorithms. After the Data Selection procedure, we are going to apply a substitution strategy for unknown feature values. Since there are lots of missing values in the original input matrix, the strategy handled this issue happens to be essential. We will use several different ways, e.g. throwing instances with missing values or imputing missing value with the most frequent value in that raw to deal with it and compare the result of them respectively. 

#### Categorical Features Encoding

We already get reasonable data that is fully filled. We are going to deal with categorical data issues. Since all of the data attributes in the datasets are categorical. And categorical data are not readable for the algorithms we select. We prefer to use one-hot encoding technique to convert the characterized feature matrix into vectorized feature space.

#### Data Normalization

Most machine learning methods are more powerful when the data are scaled into one uniform range. We will normalize the encoded feature matrix to make sure the numeric attributes range from 0 to 1.

#### Training and Test set separation

After data normalization is done, we are going to separate the whole data set into training and test set respectively. The ratio between the training and test data is 4: 1.

#### Data balance 

Since we get the training set, we will observe the ratio of different labeled instances to see if there is a class imbalance in the training set. we will handle this issue by up-sampling the minor class or down-sampling the major class.

### Training

For the training part, we are going to build a KNN classifier and a Random Forest classifier based on the training set by using a hyperparameter search with 5-fold cross-validation techniques which are supported by Scikit Learn to find the best hyperparameters building the model. Besides, based on common sense, the cost of “failing to detect a poisonous mushroom” should be much higher than “failing to recognize an edible mushroom”. In other words, it is more important to detect a poisonous mushroom. So we will use a customized cost matrix to help build the classifiers as well.

### Evaluation 

To evaluate the performance of models, F1 score and accuracy will be used as the main standard. We will compare the two classifiers with those scores by testing the test set to see which one is better. 

