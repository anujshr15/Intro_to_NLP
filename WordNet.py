from nltk.corpus import wordnet

syns=wordnet.synsets("plan")

print(syns[0].lemmas())
#
# print(syns[0].definition())
#
# print(syns[0].examples())

# synonyms=[]
# antonyms=[]
#
# for l in wordnet.synsets("good"):
#     for i in l.lemmas():
#         synonyms.append(i.name())
#         if i.antonyms():
#             antonyms.append(i.antonyms()[0].name())
#
# print(set(synonyms))
# print(set(antonyms))

w1=wordnet.synset("good.n.01")
w2=wordnet.synset("good.n.01")
print(w1.wup_similarity(w2))