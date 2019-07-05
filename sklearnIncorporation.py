import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.svm import SVC,NuSVC




documents=[]
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)),category))

random.shuffle(documents)

all_words=[]
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words=nltk.FreqDist(all_words)

word_features=list(all_words.keys())[:3000]

def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words)
    return features

#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets=[(find_features(rev),category) for (rev,category) in documents]

training_set=featuresets[:1900]
testing_set=featuresets[1900:]

#classifier=nltk.NaiveBayesClassifier.train(training_set)


classifier_f= open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_f)
classifier_f.close()


print("Original Naive Bayes accuracy: ",nltk.classify.accuracy(classifier,testing_set)*100)
#classifier.show_most_informative_features(15)


# save_classifier=open("naivebayes.pickle","wb")
# pickle.dump(classifier,save_classifier)
# save_classifier.close()


# GaussianNB,MultinomialNB,BernoulliNB
#  SGDClassifier,LogisticRegression
#  SVC,NuSVC

# GNB_classifier=SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print("GNB Naive Bayes accuracy: ",nltk.classify.accuracy(GNB_classifier,testing_set)*100)

MultiNB_classifier=SklearnClassifier(MultinomialNB())
MultiNB_classifier.train(training_set)
print("MultiNB Naive Bayes accuracy: ",nltk.classify.accuracy(MultiNB_classifier,testing_set)*100)

BernNB_classifier=SklearnClassifier(BernoulliNB())
BernNB_classifier.train(training_set)
print("BernNB Naive Bayes accuracy: ",nltk.classify.accuracy(BernNB_classifier,testing_set)*100)

SGD_classifier=SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGD accuracy: ",nltk.classify.accuracy(SGD_classifier,testing_set)*100)

LR_classifier=SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)
print("LR accuracy: ",nltk.classify.accuracy(LR_classifier,testing_set)*100)

SVC_classifier=SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC accuracy: ",nltk.classify.accuracy(SVC_classifier,testing_set)*100)

NuSVC_classifier=SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC accuracy: ",nltk.classify.accuracy(NuSVC_classifier,testing_set)*100)