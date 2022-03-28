import numpy
import random
import os
import re
import time
from wordpairdata import GetPairs
import pickle

wordpairpath_capital = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E01 [country - capital].txt'
wordpairpath_things_color = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E09 [things - color].txt'
wordpairpath_male_female = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E10 [male - female].txt'
wordpairpath_name_nation = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E04 [name - nationality].txt'
wordpairpath_antonyms_binary = 'C:\\MSCS\\cs536\\final project\data\\analogy\\BATS_3.0\\4_Lexicographic_semantics\\L10 [antonyms - binary].txt'
wordpairpath_adj_comparative = 'analogy\\BATS_3.0\\1_Inflectional_morphology\\I03 [adj - comparative].txt'
wordpairpath_animal_sound = 'C:\\MSCS\\cs536\\final project\\data\\analogy\\BATS_3.0\\3_Encyclopedic_semantics\\E07 [animal - sound].txt'
wordpairpath_animal_shelter = 'C:\\MSCS\\cs536\\final project\\data\\analogy\\BATS_3.0\\3_Encyclopedic_semantics\\E08 [animal - shelter].txt'
def SentencesExtract(word_pair_data_path):
    count=0
    pairs = []
    pairs+=GetPairs(word_pair_data_path)
    print(pairs)
    templates_raw = []
    for i in range(10):
        f = open('C:/MSCS/cs536/final project/data/wiki/wiki_0'+str(i),'rb')
        for line in f:
            sentences = line.split(b'.')
            for sentence in sentences:
                tokenlist = sentence.split()
                if len(tokenlist)<100:
                    for word1,word2 in pairs:
                        if word1 in tokenlist and word2 in tokenlist:
                            ind1 = tokenlist.index(word1)
                            ind2 = tokenlist.index(word2)
                            if abs(ind1-ind2)<=15:
                                templates_raw.append(sentence)
                                print(word1,word2,":")
                                print(sentence)
                                print()
                                count+=1
        print(count)
        f.close()
    print(len(templates_raw))
    with open('C:/MSCS/cs536/final project/data/'+word_pair_data_path[-15:-5]+'.txt', 'wb') as tempfile:
        pickle.dump(templates_raw, tempfile)


if __name__ == "__main__":
    #SentencesExtract(wordpairpath_name_nation)
    #SentencesExtract(wordpairpath_antonyms_binary)
    #SentencesExtract(wordpairpath_adj_comparative)
    #SentencesExtract(wordpairpath_animal_sound)
    SentencesExtract(wordpairpath_animal_shelter)