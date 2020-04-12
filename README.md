# SOEN691Project
## Abstract
This report will perform analysis on a dataset which contains physical characters of poisonous or edible mushrooms. The dataset is Mushroom Data Set [1] from UCI machine learning repository provided by Jeff Schlimmer. First, we will preprocess this dataset to get the training set and testing set. Then use the training set to train a random forest classifier, a k-nearest neighbors classifier, and a naive bayes classifier in Spark, and try to predict whether a mushroom is safe or poisonous with using a customized cost matrix. Second, we will analyze the results and also compare the accuracy, precision, recall, F1 score and cost of time of these techniques. 
## Introduction
Mushrooms are a common ingredient in human food. However, some kinds of mushrooms can cause severe, sometimes even fatal, food poisoning, and these accidents usually due to misidentification [2]. Therefore, this project targets to apply supervised machine learning techniques on poisonous mushroom identification. We will use a mushroom dataset from UCI Machine Learning Repository. It has physical characteristics about 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom. This data set has 22 features and two classes: edible, definitely poisonous, unknown edibility and not recommended (these two classes are combined with the poisonous one by the original data provider). It contains 8124 instances and there are missing values in some of those instances. Verma, S. K., and M. Dutta have done a similar research on the same dataset using ANN and ANFIS algorithm [3], and Muhammad Husaini has evaluated Naive Bayes, RIDOR, and Bayes Net algorithms on this dataset [4]. These two previous works both get promising results on this dataset. The accuracy scores range from 94% to 100% in these researches. Therefore, the expected training scores in our project are also high.
<br/>
To add more comparisons, this project will use this dataset to train KNN, random forest, and naive bayes classifiers and do the prediction. We will mainly use Spark since it supports parallelization to provide better training efficiency, and we will use scikit-learn to train KNN because spark doesn’t provide a KNN classifier implementation. Here are some problems we need to solve: How to select useful features? How to deal with missing values? And how to represent categorical data in numerical values to make it feasible for selected classifiers? After this, we need to consider how to apply the methods provided by Spark on the data. Then, after we get the model, we need to use the model on the test set and analyze the results. At the end, we will conclude which model can achieve a better prediction, and which features in this dataset are more important. 
## Materials and Methods
We will use the Mushroom data set from UCI to train three different classifiers based on KNN, Random Forest and Naive Bayes algorithms. We are going to divide the whole process into three individual steps: Data Preprocessing, Training and Evaluation.  
### Data Preprocessing
As soon as we get the raw data. The very first one is preprocessing the dataset. There are five step-by-step methods in this part: Dealing with missing value, Data Selection, Categorical Features Encoding, Training and Test set separation, Data Transformation.
#### Dealing with missing value
As we know, real-world datasets often contain missing values which are not usually acceptable for most machine learning algorithms. Since there are lots of missing values in the original input matrix, the strategy handled this issue happens to be essential. We will use several different ways: if the missing values take majority in a certain feature, we will drop that feature; if not, we will delete the record which contains missing values.  
#### Data Selection
There are 22 different physical features for this task while the information extracted from them varies a lot. I.e. Some of the features tell us more useful hints compared with the rest. We are going to compute the Pearson's correlation coefficients of all features and drop one feature in each strongly correlated feature pair. In addition, we will also drop the feature which has only one distinct value.
#### Categorical Features Encoding
We already get reasonable data that is fully filled. We are going to deal with categorical data issues. Since all of the data attributes in the datasets are categorical. And categorical data are not readable for the algorithms we select. We prefer to use one-hot encoding technique to convert the characterized feature matrix into vectorized feature space.
#### Training and Test set separation
After encoding the data, we are going to separate the whole data set into training and test set respectively. The ratio between the training and test data is 80%: 20%.
#### Data Transformation
Since we apply the one-hot encoding to the features, each categorical feature will be represented as a higher dimensional vector. Therefore, we also will perform PCA technique to reduce the dimension of this dataset. 
### Training
For the training part, we are going to build a KNN classifier, a Random Forest classifier and Naive Bayes classifier based on the training set by using a hyperparameter search with 5-fold cross-validation techniques to find the best hyperparameters for building the model. In addition, we will also train the same classifiers using the dataset before performing PCA reduction and to compare the results.
### Evaluation and Analysis
To evaluate the results of models, precision, recall, F1 score and accuracy will be used as the main standard. We will compare the two classifiers with those scores by testing the test set to see which one is better. In addition, we will analyze the feature importance using the Random Forest classifier.
We will also compare the time cost of training the classifiers to find which one can achieve better performance.
## Results
### Dataset without PCA reduction
| Classifier | Accuracy | Recall  | Precision | F1 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| KNN  | 100.00% | 100.00% | 100.00% | 100.00% |
| Random Forest  | 100.00% | 100.00% | 100.00% | 100.00% |
| Naive Bayes  | 99.81% |  99.81% | 99.82%  | 99.81% |
![KNN without PCA](https://github.com/xwang1109/SOEN691Project/blob/master/output/knnfeatures.png)
![Random Forest without PCA](https://github.com/xwang1109/SOEN691Project/blob/master/output/Random%20Forest%20Classification%20with%20features.png)
![Naive Bayes without PCA](https://github.com/xwang1109/SOEN691Project/blob/master/output/Naive%20Bayes%20Classification%20with%20features.png)
### Dataset with PCA reduction
| Classifier | Accuracy | Recall  | Precision | F1 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| KNN  | 97.77% | 96.28% | 99.08% | 97.66% |
| Random Forest  | 97.71% | 97.71% | 97.73% | 97.71% |
| Naive Bayes  | 88.45% |  88.45% | 90.05%  | 88.28% |
![KNN with PCA](https://github.com/xwang1109/SOEN691Project/blob/master/output/knnpca_features.png)
![Random Forest with PCA](https://github.com/xwang1109/SOEN691Project/blob/master/output/Random%20Forest%20Classification%20with%20pca_features.png)
![Naive Bayes with PCA](https://github.com/xwang1109/SOEN691Project/blob/master/output/Naive%20Bayes%20Classification%20with%20scaledPCAFeatures.png)
## Reference
[1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.<br/>
[2] Lima, A. D., Fortes, R. C., Novaes, M. G., & Percário, S. (2012). Poisonous mushrooms; a review of the most common intoxications. Nutricion hospitalaria, 27(2), 402-408.<br/>
[3] Verma, S. K., & Dutta, M. (2018). Mushroom classification using ANN and ANFIS algorithm. IOSR Journal of Engineering (IOSRJEN), 8(01), 94-100.<br/>
[4] Husaini, Muhammad. "A Data Mining Based On Ensemble Classifier Classification Approach for Edible Mushroom Identification." (2018).
