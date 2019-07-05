import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_raw=state_union.raw("2005-GWBush.txt")
sample_raw=state_union.raw("2006-GWBush.txt")

tokenizer=PunktSentenceTokenizer(train_raw)
sentences=tokenizer.tokenize(sample_raw)

def process_data():
    try:
        for sentence in sentences:
            words=nltk.word_tokenize(sentence)
            tagged=nltk.pos_tag(words)
            ChunkGram=r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser=nltk.RegexpParser(ChunkGram)
            chunked=chunkParser.parse(tagged)
            chunked.draw()

    except Exception as e:
        print(str(e))

process_data()

