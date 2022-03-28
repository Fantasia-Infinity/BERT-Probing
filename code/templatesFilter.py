import random
import pickle
import copy
import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from wordpairdata import GetPairs
import copy
import csv

wordpairpath_capital = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E01 [country - capital].txt'
wordpairpath_things_color = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E09 [things - color].txt'
wordpairpath_male_female = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E10 [male - female].txt'
wordpairpath_name_nation = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E04 [name - nationality].txt'
wordpairpath_antonyms_binary = 'C:\\MSCS\\cs536\\final project\data\\analogy\\BATS_3.0\\4_Lexicographic_semantics\\L10 [antonyms - binary].txt'
wordpairpath_animal_sound = 'C:\\MSCS\\cs536\\final project\\data\\analogy\\BATS_3.0\\3_Encyclopedic_semantics\\E07 [animal - sound].txt'
wordpairpath_animal_shelter = 'C:\\MSCS\\cs536\\final project\\data\\analogy\\BATS_3.0\\3_Encyclopedic_semantics\\E08 [animal - shelter].txt'

sentencdump_things_color_path = 'C:/MSCS/cs536/final project/data/templates-thing-color.txt'
sentencdump_male_female_path = 'C:/MSCS/cs536/final project/data/templates-male-female.txt'
sentencdump_antonyms_binary_path = 'templates-antonyms-binary.txt'
sentencdump_animal_sound_path = 'templates-animal-sound.txt'
sentencdump_animal_shelter_path = 'templates- animal-shelter.txt'
class BertForFilter(nn.Module):
    def __init__(self):
        super(BertForFilter, self ).__init__() 
        #self.tokenizer = BertTokenizer.from_pretrained('C:/Alchemy/bert_model/bert-base-uncased/bert-base-uncased-vocab.txt')
        self.encoder = BertModel.from_pretrained('C:/Alchemy/bert_model/bert-base-uncased/')
        self.linear = nn.Linear(768,30522)
        self.linear.weight.data = self.encoder.embeddings.word_embeddings.weight
    def forward(self,input_inds_list,word_mask,ind_pred,k=None):#[1,len,]
        if k == None:
            bertoutput =  self.encoder(input_ids=torch.tensor([input_inds_list]).to('cuda'),attention_mask=word_mask.to('cuda'))
            outputembeddings = bertoutput[0]
            logits_pred = self.linear(outputembeddings)
            maxpred = torch.argmax(logits_pred,dim = 2).squeeze()
            predindlist = list(maxpred.squeeze())
            return predindlist[ind_pred]
        else:
            bertoutput =  self.encoder(input_ids=torch.tensor([input_inds_list]).to('cuda'),attention_mask=word_mask.to('cuda'))
            outputembeddings = bertoutput[0] #[batch,len,dim]
            target_ind_emb = outputembeddings[:,ind_pred,:] #[batch,dim]
            target_logits_pred = self.linear(target_ind_emb) #[batch,vocab]

            #maxpred = torch.argmax(logits_pred,dim = 2).squeeze()
            #predindlist = list(maxpred.squeeze())
            #return predindlist[ind_pred]
            topkpredids = torch.topk(target_logits_pred,k)[1] #indices
            #print('topkpredids.shape:',topkpredids.shape)
            return list(topkpredids.squeeze()) #list of topk ids

tokenizer = BertTokenizer.from_pretrained('C:/Alchemy/bert_model/bert-base-uncased/bert-base-uncased-vocab.txt')
MLMmodel = BertForFilter()

def FasterScoreSentence(sentence,word_pairs,k):
    tokenizer = BertTokenizer.from_pretrained('C:/Alchemy/bert_model/bert-base-uncased/bert-base-uncased-vocab.txt')
    MLMmodel = BertForFilter()
    MLMmodel.to('cuda')

    encode = tokenizer(sentence)
    input_ids = encode['input_ids'] #[1,len]
    list_input_inds = input_ids
    #print('list input inds:',list_input_inds)
    def find_sentence_holes(input_ids,wordpairs):
        input_token_list = tokenizer.convert_ids_to_tokens(input_ids)
        #print('input_token_list:',input_token_list)
        for word1,word2 in word_pairs:
            word1 = str(word1,encoding='utf-8')
            word2 = str(word2,encoding='utf-8')
            #print(word1,word2)
            if word1 in input_token_list and word2 in input_token_list:
                hole1_ind = input_token_list.index(word1)
                hole2_ind = input_token_list.index(word2)
                #print('hole1:',hole1_ind,'hole2:',hole2_ind)
                return hole1_ind,hole2_ind

    hole1_ind,hole2_ind = find_sentence_holes(input_ids,word_pairs)
    hole1_attention_mask = torch.ones_like(torch.tensor([input_ids]))
    hole1_attention_mask[:,hole1_ind]=0
    hole2_attention_mask = torch.ones_like(torch.tensor([input_ids]))
    hole2_attention_mask[:,hole2_ind]=0

    hole1_topk_pred_ind = MLMmodel(list_input_inds,hole1_attention_mask,hole1_ind,k)
    hole2_topk_pred_ind = MLMmodel(list_input_inds,hole2_attention_mask,hole2_ind,k)
    count = 0
    for word1,word2 in word_pairs:
        word1 = str(word1,encoding='utf-8')
        word2 = str(word2,encoding='utf-8')
        word1_ind = tokenizer(word1)['input_ids'][1]#.squeeze().item() #int
        word2_ind = tokenizer(word2)['input_ids'][1]
        for word1_pred_ind in hole1_topk_pred_ind:
            if word1_ind == word1_pred_ind:
                count+=1
        
        for word2_pred_ind in hole2_topk_pred_ind:
            if word2_ind == word2_pred_ind:
                count+=1
        
    return count,hole1_ind,hole2_ind


def ScoreSentence(sentence,word_pairs):#(string sentetnce,list of word pairs)
    tokenizer = BertTokenizer.from_pretrained('C:/Alchemy/bert_model/bert-base-uncased/bert-base-uncased-vocab.txt')
    MLMmodel = BertForFilter()
    MLMmodel.to('cuda')

    encode = tokenizer(sentence)
    input_ids = encode['input_ids'] #[1,len]
    list_input_inds = input_ids
    #print('list input inds:',list_input_inds)
    
    def find_sentence_holes(input_ids,wordpairs):
        input_token_list = tokenizer.convert_ids_to_tokens(input_ids)
        #print('input_token_list:',input_token_list)
        for word1,word2 in word_pairs:
            word1 = str(word1,encoding='utf-8')
            word2 = str(word2,encoding='utf-8')
            #print(word1,word2)
            if word1 in input_token_list and word2 in input_token_list:
                hole1_ind = input_token_list.index(word1)
                hole2_ind = input_token_list.index(word2)
                #print('hole1:',hole1_ind,'hole2:',hole2_ind)
                return hole1_ind,hole2_ind

    #print(find_sentence_holes(input_ids,word_pairs))
    hole1_ind,hole2_ind = find_sentence_holes(input_ids,word_pairs)
    count = 0
    for word1,word2 in word_pairs:
        word1 = str(word1,encoding='utf-8')
        word2 = str(word2,encoding='utf-8')
        #print("word1:",word1)
        word1_ind = tokenizer(word1)['input_ids'][1]#.squeeze().item() #int
        word2_ind = tokenizer(word2)['input_ids'][1]
        #print('word1_ind:',word1_ind)

        temp_input_inds = copy.deepcopy(input_ids)
        temp_input_inds[hole2_ind] = word2_ind

        attention_mask = torch.ones_like(torch.tensor([temp_input_inds]))
        attention_mask[:,hole1_ind]=0

        pred_word_ind = MLMmodel(temp_input_inds,attention_mask,hole1_ind)
        if pred_word_ind == word1_ind:
            count+=1

    for word1,word2 in word_pairs:
        word1 = str(word1,encoding='utf-8')
        word2 = str(word2,encoding='utf-8')
        #print("word1:",word1)
        word1_ind = tokenizer(word1)['input_ids'][1]#.squeeze().item() #int
        word2_ind = tokenizer(word2)['input_ids'][1]
        #print('word1_ind:',word1_ind)

        temp_input_inds = copy.deepcopy(input_ids)
        temp_input_inds[hole1_ind] = word1_ind

        attention_mask = torch.ones_like(torch.tensor([temp_input_inds]))
        attention_mask[:,hole1_ind]=0

        pred_word_ind = MLMmodel(temp_input_inds,attention_mask,hole2_ind)
        if pred_word_ind == word2_ind:
            count+=1

    return count,hole1_ind,hole2_ind

def FasterScoreSentenceAndSave(wordpair_path,sentencesdump_path):
    wordpairs = GetPairs(wordpair_path)
    f = open(sentencesdump_path, 'rb')
    sentences = pickle.load(f) #list of raw b'sentences
    print("num of sentences total:",len(sentences))
    processcount = 0

    file=open('fastscoretemp_'+wordpair_path[-15:-5]+'.csv','a',newline='')
    file_write=csv.writer(file)

    for sentence in sentences:
        try:
            sentence = str(sentence,encoding='utf-8')
            rate,hole1_ind,hole2_ind = FasterScoreSentence(sentence,wordpairs,20)
            print('sent num:',processcount)
            print('sentence:',sentence)
            print('rate:',rate)
            file_write.writerow([sentence,rate,hole1_ind,hole2_ind,processcount])
            processcount+=1
            print()
        except:
            pass
    
    file.close()

def ScoreSentenceAndSave(wordpair_path,sentencesdump_path):
    wordpairs = GetPairs(wordpair_path)
    f = open(sentencesdump_path, 'rb')
    sentences = pickle.load(f) #list of raw b'sentences
    print("num of sentences total:",len(sentences))
    processcount = 0

    file=open('scoretemp_'+wordpair_path[-15:-5]+'.csv','a',newline='')
    file_write=csv.writer(file)

    for sentence in sentences:
        '''
        sentence = str(sentence,encoding='utf-8')
        rate,hole1_ind,hole2_ind = ScoreSentence(sentence,wordpairs)
        print('sent num:',processcount)
        print('sentence:',sentence)
        print('rate:',rate)
        file_write.writerow([sentence,rate,hole1_ind,hole2_ind,processcount])
        processcount+=1
        print()
        '''
        
        try:
            sentence = str(sentence,encoding='utf-8')
            rate,hole1_ind,hole2_ind = ScoreSentence(sentence,wordpairs)
            print('sent num:',processcount)
            print('sentence:',sentence)
            print('rate:',rate)
            file_write.writerow([sentence,rate,hole1_ind,hole2_ind,processcount])
            processcount+=1
            print()
        except:
            pass
        
    file.close()
'''
def testScoreing():
    wordpairs = GetPairs(wordpairpath_male_female)
    f = open('C:/MSCS/cs536/final project/data/templates-male-female.txt', 'rb')
    sentences = pickle.load(f) #list of raw b'sentences
    print("num of sentences total:",len(sentences))
    processcount = 0
    for sentence in sentences:
        sentence = str(sentence,encoding='utf-8')
        rate = ScoreSentence(sentence,wordpairs)
        print('sent num:',processcount)
        print('sentence:',sentence)
        print('rate:',rate)
        processcount+=1
        print()

def testFasterScoreing():
    wordpairs = GetPairs(wordpairpath_male_female)
    f = open('C:/MSCS/cs536/final project/data/templates-male-female.txt', 'rb')
    sentences = pickle.load(f) #list of raw b'sentences
    print("num of sentences total:",len(sentences))
    processcount = 0
    for sentence in sentences:
        sentence = str(sentence,encoding='utf-8')
        rate = FasterScoreSentence(sentence,wordpairs,20)
        print('sent num:',processcount)
        print('sentence:',sentence)
        print('rate:',rate)
        processcount+=1
        print()
'''     

def Filter(tempscorefile,num_of_ret):
    pass


if __name__ == "__main__":
    #testScoreing()
    #testFasterScoreing()
    #FasterScoreSentenceAndSave(wordpairpath_things_color,sentencdump_things_color_path)
    #FasterScoreSentenceAndSave(wordpairpath_male_female,sentencdump_male_female_path)
    #ScoreSentenceAndSave(wordpairpath_male_female,sentencdump_male_female_path)
    #FasterScoreSentenceAndSave(wordpairpath_antonyms_binary,sentencdump_antonyms_binary_path)
    #FasterScoreSentenceAndSave(wordpairpath_animal_sound,sentencdump_animal_sound_path)
    FasterScoreSentenceAndSave(wordpairpath_animal_shelter,sentencdump_animal_shelter_path)