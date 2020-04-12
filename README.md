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
For the training part, we are going to build a KNN classifier, a Random Forest classifier and Naive Bayes classifier based on the training set by using a hyperparameter search with 5-fold cross-validation techniques to find the best hyperparameters for building the model. Since we transfer the oringinal categorical data to vectors, we will use the distance between vectors as the distance in determination of the k nearest neighbors. In addition, we will also train the same classifiers using the dataset before performing PCA reduction and to compare the results.
### Evaluation and Analysis
To evaluate the results of models, precision, recall, F1 score and accuracy will be used as the main standard. We will compare the two classifiers with those scores by testing the test set to see which one is better. In addition, we will analyze the feature importance using the Random Forest classifier.
We will also compare the time cost of training the classifiers to find which one can achieve better performance.
## Results

### Correlation Matrix
<img title="correlation matrix" src="https://github.com/xwang1109/SOEN691Project/blob/master/output/correlation_matrix_new.png" width=70%/>
From the Pearson's correlation matrix, we can find that gill-attachment and veil-color are strongly correlated. Therefore we can drop one of them to train the classifier to reduce overhead. 

### Feature Importances
<img title="feature importance" src="https://github.com/xwang1109/SOEN691Project/blob/master/output/feature_imp.png"/>
The feature importances graph shows that odor and spore-print-color are two most important features to classify an edible/poisonous mushroom, while veil-color and gill-attachment have little influence on the classification. 
### Dataset without PCA reduction

Classifier | Accuracy | Recall  | Precision | F1 | Best Hyperparameters
------------- | ------------- | ------------- | ------------- | ------------- |---
KNN  | 100.00% | 100.00% | 100.00% | 100.00% | number of neighbors = 10
Random Forest  | 100.00% | 100.00% | 100.00% | 100.00% | maxDepth = 5, NumTrees = 10
Naive Bayes  | 99.81% |  99.81% | 99.82%  | 99.81% | Smoothing = 0
Time cost | 42 seconds

<img title="KNN without PCA" src="https://github.com/xwang1109/SOEN691Project/blob/master/output/knnfeatures.png" width=70%/>
<img title="Random Forest without PCA" src="https://github.com/xwang1109/SOEN691Project/blob/master/output/Random%20Forest%20Classification%20with%20features.png" width=70%/>
<img title="Naive Bayes without PCA" src="https://github.com/xwang1109/SOEN691Project/blob/master/output/Naive%20Bayes%20Classification%20with%20features.png" width=70%/>
<br/>
Using dataset with original features after one-hot enconding, the KNN and Random Forest can reach 100% accuracy and F1 score, which is better than Naive Bayes.

### Dataset with PCA reduction
Classifier | Accuracy | Recall  | Precision | F1 | Best Hyperparameters
------------- | ------------- | ------------- | ------------- | ------------- |---
KNN  | 97.77% | 96.28% | 99.08% | 97.66% | number of neighbors = 25
Random Forest  | 97.71% | 97.71% | 97.73% | 97.71% | maxDepth = 5, NumTrees = 10
Naive Bayes  | 88.45% |  88.45% | 90.05%  | 88.28% | Smoothing = 0
Time cost | 34 seconds

<img title="KNN with PCA" src="https://github.com/xwang1109/SOEN691Project/blob/master/output/knnpca_features.png" width=70%/>
<img title="Random Forest with PCA" src="https://github.com/xwang1109/SOEN691Project/blob/master/output/Random%20Forest%20Classification%20with%20pca_features.png" width=70%/>
<img title="Naive Bayes with PCA" src="https://github.com/xwang1109/SOEN691Project/blob/master/output/Naive%20Bayes%20Classification%20with%20scaledPCAFeatures.png" width=70%/>
<br/>
Using dataset with PCA redunction features, the scores are all lower compared with 3 classifiers than the previous case, but the trainning time cost is also lower. The naive bayes classifier is most influenced by the PCA reduction: the scores significantly decrease by 10%. That is because PCA features have negative values, but naive bayes can only deal with non-negative values. So we have to apply a min-max scaler to rescale the features, which will influence the characteristics of the data. The results also show that KNN and Random Forest classifiers can achieve better trainning results than Naive Bayes.
## Discussion
In the project, we mainly applied feature selection, feature transformation and three classifiers on the dataset. We compared the trainning results of knn, random forest and naive bayes, also we compared the trainning results before and after applying PCA feature reduction technique. To conclude, we find:<br/>
1. Random forest and knn models provide the best prediction for this dataset, while Naive Bayes model’s accuracy and f1 score are relatively lower than other two models, but it can still provide a good result. <br/>
2. The training time cost for data applied PCA is lower than the original 83-dimensional data (34 seconds vs. 42 seconds), however, the trade-off is lower accuracy and f1 score. <br/>
3. The odor and spore-print-color siginificantly influence the classification of the edible/poisonous mushroom.<br/>
The limitation of this research is mainly due to the characteristic of this dataset. Since this dataset is well organized and with few noisy values, it is easy to get good results. In the future work, we can perform the similar techniques on more complicated data to see the effects. On the other hand, from this research we can conclude that a mushroom’s edibility is high related with its physical features. However, in this dataset, these features have already been extracted and listed manually, which is very time consuming. In the future, we can use photos of different mushrooms to train  a CNN to do the edible/poisonous classification to reduce human work. 
 


## Reference
[1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.<br/>
[2] Lima, A. D., Fortes, R. C., Novaes, M. G., & Percário, S. (2012). Poisonous mushrooms; a review of the most common intoxications. Nutricion hospitalaria, 27(2), 402-408.<br/>
[3] Verma, S. K., & Dutta, M. (2018). Mushroom classification using ANN and ANFIS algorithm. IOSR Journal of Engineering (IOSRJEN), 8(01), 94-100.<br/>
[4] Husaini, Muhammad. "A Data Mining Based On Ensemble Classifier Classification Approach for Edible Mushroom Identification." (2018).
