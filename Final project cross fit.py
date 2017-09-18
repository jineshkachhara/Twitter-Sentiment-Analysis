"""
A simple script that demonstrates how we classify textual data with sklearn.

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
import operator
import codecs


#read the reviews and their polarities from a given file
def loadData(fname):
    reviews=[]
    labels=[]
    f=codecs.open(fname,encoding='utf-8')
    for line in f:
        temp=line.strip().split('\t')
        reviews.append(temp[0].lower())
        labels.append(int(temp[1]))
    f.close()
    return reviews,labels

def feature(feature_name,feature_importance):
    TopFeature=[]
    TopValue=[]
    features_dict={}   
    for name,importance in list(zip(feature_name,feature_importance)):
        if features_dict.get(name) == None:
            features_dict[name] = importance   
        #else: features_dict[name] += importance           
    for w in sorted(features_dict.items(),key=operator.itemgetter(1),reverse=True):         
        TopFeature.append(w[0])
        TopValue.append(w[1])
    return TopFeature,TopValue

rev_train,labels_train=loadData('tweets_trump_train.txt')
rev_test,labels_test=loadData('tweets_trump_test.txt')

#Build a counter based on the training dataset
counter = CountVectorizer(stop_words=stopwords.words('english'))
counter.fit(rev_train)
feature_names = counter.get_feature_names()
#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data

#Applying multiple classifiers
KNN_classifier=KNeighborsClassifier()
LREG_classifier=LogisticRegression()
DT_classifier = DecisionTreeClassifier()
MLP_Classifier = MLPClassifier()
SVM_Classifier = LinearSVC()
RF_classifier = RandomForestClassifier()

#RF_classifier = RandomForestClassifier(n_estimators=100,max_features='auto',random_state=20)
#RF_classifier = ExtraTreesClassifier(n_estimators=100, random_state = 20)
#predictors=[('knn',KNN_classifier),('lreg',LREG_classifier),('dt',DT_classifier)]

predictors=[('knn',KNN_classifier),('SVM',SVM_Classifier),('rf',RF_classifier),('MLP',MLP_Classifier),('lreg',LREG_classifier)]
#predictors=[('knn',KNN_classifier),('rf',RF_classifier)]

VT=VotingClassifier(predictors)

'''
#=======================================================================================
#build the parameter grid for KNeighbors classifier
#KNN_grid = [{'n_neighbors': [1,3,5,7,9,11,13,15,17], 'weights':['uniform','distance']}]
KNN_grid = [{'n_estimators': [100,150,200,250], 'max_features':['auto','sqrt','log2']}]

#build a grid search to find the best parameters
gridsearchKNN = GridSearchCV(RF_classifier, KNN_grid, cv=5)

#run the grid search
gridsearchKNN.fit(counts_train,labels_train)


VT.fit(counts_train,labels_train)

#use the VT classifier to predict
predicted=VT.predict(counts_test)

#print the accuracy
print (accuracy_score(predicted,labels_test))
'''
#=======================================================================================
#build the parameter grid for Random Forest Classifier
RF_grid = [{'n_estimators': [100,150,200,250], 'random_state':[10,20,25]}]


#build a grid search to find the best parameters
gridsearchRF = GridSearchCV(RF_classifier, RF_grid, cv=5)

#run the grid search
gridsearchRF.fit(counts_train,labels_train)

VT.fit(counts_train,labels_train)

#use the VT classifier to predict
predicted=VT.predict(counts_test)

#print the accuracy
print (accuracy_score(predicted,labels_test))
'''
==========================================================================================================
# List of classifiers and their corresponding accuracies that were individually applied on sample dataset.
==========================================================================================================
clf = KNeighborsClassifier(weights='distance',algorithm='brute') Accuracy- 0.840531561462
clf = ExtraTreesClassifier(n_estimators=250, random_state = 0) Accuracy- 0.833887043189
clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False) Accuracy - 0.813953488372
clf = KNeighborsClassifier(weights='distance',algorithm='brute',p=1) Accuracy- 0.812292358804
clf = BaggingClassifier() #0.8073089701
clf = RandomForestClassifier(random_state=0) Accuracy- 0.808970099668
clf = LinearSVC(loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=0, max_iter=1000) Accuracy- 0.802325581395
clf = AdaBoostClassifier(base_estimator=None, n_estimators=890, learning_rate=1.0, algorithm='SAMME.R', random_state=1) Accuracy- 0.800664451827
clf = MLPClassifier() Accuracy- 0.78903654485
clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None) Accuracy - 0.737541528239
clf = KNeighborsClassifier() Accuracy - 0.639534883721
clf = BaggingClassifier(KNeighborsClassifier(),max_samples=1000, random_state = 0) Accuracy - 0.611295681063
'''
# For printitng the graph of top 10 important features
RF_classifier.fit(counts_train,labels_train)
imp = RF_classifier.feature_importances_
TopFeature,TopValue = feature(feature_names,imp)
std = np.std([tree.feature_importances_ for tree in RF_classifier.estimators_],axis=0)
indices = np.argsort(imp)[::-1]
l = len(indices)
print("Feature ranking:")
for f in range(0,10):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], TopFeature[f], imp[indices[f]]))
#for feature in zip(, RF_classifier.feature_importances_):
#    print(feature)
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(0,10), imp[indices][:10],color="r", yerr=std[indices[:10]], align="center")
plt.xticks(range(0,10), indices)
plt.xlim([-1, 10])
plt.show()
