# Introduction
This document explores the forward algorithm in conjunction with the leave-one-out cross-validation performed on k-nearest-neighbors classifier. My program is written in Python 3 and is included in the .ipynb notebook file that accompanies this document.

# KNN
The k-nearest-neighbors algorithm works as such: given a test instance and a training set, the algorithm calculates the euclidean distance between the test instance and every training instance. It then picks the k nearest training instances to the test instance and makes a classification based on the most popular neighbor.

# Leave-One-Out Cross-Validation (CV)
The leave-one-out cross-validation works as such: it takes in a complete dataset and a set of features. Then, it pulls out a single instance (hence leave-one-out) our of the dataset. After, it calls on the KNN algorithm using the smaller dataset as training data and the singled out instance as a test data. Since the cross-validation function knows the label of the single test instance, it can judge whether the KNN was accurate or not. It stores the accuracy of n KNN calls where n is the number of dataset instances, and then returns a mean value of these accuracies.

# Feature Search
Feature search works as such: it takes subsets of the dataset features and calls the leave-one-out cross-validation algorithm. It stores the mean value of the accuracies returned by the CV and stores it. When all feature combinations are exhausted, it returns the combination with the highest mean accuracy value. This combination is the best subset of features that we can use on our data in the KNN classifier (all of this is done because the KNN classifier is sensitive to noise and irrelevant features).

# Design
I created four functions: KNN, cross-validator, feature search, and main. The functions are pretty straightforward. The KNN uses numpy.linalg.norm to calculate the euclidean distance of test data with every other training data. Once it has a list of these distances, it attaches the training labels to them, and sorts them based on distance in ascending order. It then picks the k most common label in the distance:label pairs and returns it.

The cross-validator pulls out a single data instance and its label from the dataset and passes it as test data to the KNN predictor. It also passes the remainder of the data to the KNN as training data. If the KNN classifier correctly predicts the label, we add 1 to our accuracy list, 0 otherwise. By the end, we calculate the mean of the accuracy list and return it.

The feature search function goes through subsets of features and for each subset, calculates the mean accuracy using the cross validator. By the end, it returns the subset with the most accuracy.

The main function is responsible for showing relevant messages and receiving your input. I have implemented default values so that you do not have to input a value every time you test/run the program.

# Statistics and Analysis
The large datasets each take about an hour and fourty minutes to run to completion. For that reason, I will only discuss the statistics of the small datasets. The following results are achieved using the z-score normalization on each dataset.

Here is the results for k = 1, 3, 5 for small 67 and 80:

Small 80:<br />
Using all 10 features, the accuracy of the 1NN is 65%<br />
k = 1: Most accurate subset is [4 , 2 ] with an accuracy of 92.0%<br />
k = 3: Most accurate subset is [4, 2, 6] with an accuracy of 90.0%<br />
k = 5: Most accurate subset is [4, 2, 7] with an accuracy of 92.0%<br />

Small 67:<br />
Using all 10 features, the accuracy of the 1NN is 81%<br />
k = 1: Most accurate subset is [0, 9 ] with an accuracy of 93.0%<br />
k = 3: Most accurate subset is [0, 9 ] with an accuracy of 94.0%<br />
k = 5: Most accurate subset is [0, 5, 7] with an accuracy of 93.0%<br />

Large 80:<br />
Using all 40 features, the accuracy of the 1NN is 69.5%<br />
k = 1: Most accurate subset is [26, 0] with an accuracy of 95.5%<br />

large 67:<br />
Using all 40 features, the accuracy of the 1NN is 66.6%<br />
k = 1: Most accurate subset is [10, 11] with an accuracy of 96.7%<br />

It is evident that features 2 and 4 contribute the most to the small 80 dataset classification. For k = 1 and 3, features 0 and 9 contribute the most to the small 67 dataset classifier.

The small 67 gets better accuracy results on average for k = 1, 3, and 5 compared to small 80

![Accuracy vs. K chart](/images/chart.png "Accuracy vs. K chart")

# Conclusion
We can observe that by not using the right subset of features, we achieve a suboptimal accuracy from our KNN classifier. This proves that KNNs are susceptible to noisy features. Using feature search, we can narrow down a large set of features (in this case up to 40) down to just 2 or 3. Doing so increases our accuracy from about 65-70% up to about 95% which is a considerable increase.
