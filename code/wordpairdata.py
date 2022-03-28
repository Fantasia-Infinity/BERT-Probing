import time
import random
import os
import re

wordpairpath_capital = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E01 [country - capital].txt'
wordpairpath_things_color = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E09 [things - color].txt'
def GetPairs(path):
    f = open(path,'rb')
    count=0
    pairlist = []
    for line in f:
        pair = line.split()
        wordpair = (pair[0],pair[1])
        #pairlist.append(wordpair)
        #count+=1
        for second in pair[1].split(b'/'):
            a_pair = (pair[0],second)
            pairlist.append(a_pair)
            count+=1       
            print(a_pair)     
    print(count)
    f.close()
    return pairlist

if __name__ == "__main__":
    print(GetPairs(wordpairpath_things_color))
