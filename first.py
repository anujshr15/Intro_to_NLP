from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# example_text="hello Anuj! How are you man? It's great to be back here. What are you doing today? Let's catch up!"
# stop_words=set(stopwords.words('english'))
# words=word_tokenize(example_text)
# filtered_sent=[]
# for w in words:
#     if w not in stop_words:
#         filtered_sent.append(w)
#
# print(filtered_sent)

stemmer=PorterStemmer()
words=["anuj","anujs","anujing","anujed"]
for w in words:
    print(stemmer.stem(w))