"""
A simple script that demonstrates how we classify textual data with sklearn.

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from nltk.corpus import stopwords
from operator import itemgetter

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
    for w in sorted(features_dict.items(),key=itemgetter(1),reverse=True):         
        TopFeature.append(w[0])
        TopValue.append(w[1])
    #for i in range(0,10):
     #   print(TopFeature[i],TopValue[i])
    return TopFeature,TopValue
        
               
rev_train,labels_train=loadData('tweets_hillary_train.txt')
rev_test,labels_test=loadData('tweets_hillary_test.txt')

#x, y = make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2, random_state=0, shuffle=False)
#Build a counter based on the training dataset
counter = CountVectorizer(stop_words=stopwords.words('english'))
counter.fit(rev_train)
feature_name=counter.get_feature_names()


#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data

#train classifier
#clf = LinearSVC(loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=0, max_iter=1000) #0.802325581395
#clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False) #0.813953488372
#clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None) #0.737541528239

#clf = RandomForestClassifier(random_state=0) #0.808970099668
#clf = AdaBoostClassifier(base_estimator=None, n_estimators=890, learning_rate=1.0, algorithm='SAMME.R', random_state=1) #0.800664451827
#clf = MLPClassifier() #0.78903654485
#clf = KNeighborsClassifier() #0.639534883721
#clf = KNeighborsClassifier(weights='distance',algorithm='brute') #0.840531561462
#clf = KNeighborsClassifier(weights='distance',algorithm='brute',p=1) #0.812292358804

#clf = BaggingClassifier() #0.8073089701
#clf = BaggingClassifier(KNeighborsClassifier(),max_samples=1000) #0.598006644518
#clf = ExtraTreesClassifier() #0.832225913621
clf = ExtraTreesClassifier(n_estimators=250, random_state = 0) #0.833887043189


#train all classifier on the same datasets
clf.fit(counts_train,labels_train)
imp = clf.feature_importances_
#print(clf.feature_importances_)
TF,TV = feature(feature_name,imp)
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(imp)[::-1]

#use hard voting to predict (majority voting)
pred=clf.predict(counts_test)

#print accuracy
print (accuracy_score(pred,labels_test))
#print (imp)
#print(counts_train.shape[1])

print("Feature ranking:")

for f in range(0,10):
    print("%d. feature %d- %s (%f)" % (f + 1, indices[f],TF[f], imp[indices[f]]))
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(0,10), imp[indices[:10]],
       color="r", yerr=std[indices[:10]], align="center")
plt.xticks(range(0,10), indices)
plt.xlim([-1, 10])
plt.show()
