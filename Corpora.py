from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
sample=gutenberg.raw("bible-kjv.txt")
sent=sent_tokenize(sample)
print(sent[5:15])
