import random
import pickle
import copy
import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from wordpairdata import GetPairs
import copy
from datetime import datetime
import csv
import pandas as ps
from torch.utils.data import random_split, Subset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

split_rate = 0.5

def set_seed(seed):  
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

wordpairpath_capital = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E01 [country - capital].txt'
wordpairpath_things_color = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E09 [things - color].txt'
wordpairpath_male_female = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E10 [male - female].txt'
wordpairpath_name_nation = 'C:/MSCS/cs536/final project/data/analogy/BATS_3.0/3_Encyclopedic_semantics/E04 [name - nationality].txt'
wordpairpath_antonyms_binary = 'C:\\MSCS\\cs536\\final project\data\\analogy\\BATS_3.0\\4_Lexicographic_semantics\\L10 [antonyms - binary].txt'
wordpairpath_animal_shelter = 'C:\\MSCS\\cs536\\final project\\data\\analogy\\BATS_3.0\\3_Encyclopedic_semantics\\E08 [animal - shelter].txt'

scoretemp_things_color_path = 'fastscoretemp_things - color.csv'
scoretemp_male_female_path = 'scoretemp_male - female.csv'
scoretemp_antonyms_binary = 'fastscoretemp_antonyms - binary.csv'
scoretemp_animal_shelter = 'fastscoretemp_animal - shelter.csv'

tokenizer = BertTokenizer.from_pretrained('C:/Alchemy/bert_model/bert-base-uncased/bert-base-uncased-vocab.txt')

def sentence_to_listofids(sentence):
    encode = tokenizer(sentence)
    input_ids = encode['input_ids']
    list_input_inds = input_ids
    return list_input_inds 

class MyDataSet(Dataset):
    def __init__(self,scoretemp_path,wordpair_path,dump=None,k=None):
        if k == None:
            self.word_pairs = GetPairs(wordpair_path)
            self.word_pairs = self.word_pairs[0:int(split_rate*len(self.word_pairs))]

            self.templates = []      #(k of sentence,hole1,hole2,score)
            self.instances = []#  list of (sentence,label)
            self.maxlen = 0
            file=open(scoretemp_path,'r') 
            file_content=csv.reader(file)
            for row in file_content:############################template 的 topk
                if row == []:
                    pass
                else:
                    #print(row)
                    sentence,score,hole1_ind,hole2_ind,_ = row
                    self.instances+=self.makeSamples(sentence,int(hole1_ind),int(hole2_ind),self.word_pairs)
            for sentenceids,label in self.instances:
                thelen = len(sentenceids)
                if thelen>self.maxlen:
                    self.maxlen = thelen
        else:
            list_of_temp = []
            self.word_pairs = GetPairs(wordpair_path)
            self.instances = []#  list of (sentence,label)
            self.templates = []#  list of (sentece,hole1,hole2,score)
            self.maxlen = 0
            file=open(scoretemp_path,'r') 
            file_content=csv.reader(file)
            for row in file_content:
                if row == []:
                    pass
                else:
                    list_of_temp.append(row)
                    #print(row)
                    #sentence,score,hole1_ind,hole2_ind,_ = row
                    #self.instances+=self.makeSamples(sentence,int(hole1_ind),int(hole2_ind),self.word_pairs)
            list_of_temp.sort(key=lambda row:int(row[1]),reverse=True)
            for i in range(k):
                sentence,score,hole1_ind,hole2_ind,_ = list_of_temp[i]
                print(sentence,score)
                self.templates.append((sentence,hole1_ind,hole2_ind,score))
                self.instances+=self.makeSamples(sentence,int(hole1_ind),int(hole2_ind),self.word_pairs)        
            for sentenceids,label in self.instances:
                thelen = len(sentenceids)
                if thelen>self.maxlen:
                    self.maxlen = thelen    

        

        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self,id):# return (tensor[maxlen],int,tnesor[maxlen])
        raw_senteceids,label = self.instances[id]
        raw_senteceids_tensor = torch.tensor(raw_senteceids)
        lenght = len(raw_senteceids)
        attention_mask = torch.ones_like(raw_senteceids_tensor)
        sentenceids_tensor = torch.cat([raw_senteceids_tensor, torch.zeros(self.maxlen - len(raw_senteceids))])
        attention_mask = torch.LongTensor(torch.cat([attention_mask, torch.zeros(self.maxlen - lenght)]).numpy())######
        label = torch.LongTensor([label])
        return sentenceids_tensor,label,attention_mask
    
  

    def makeSamples(self,sentence,hole1_ind,hole2_ind,wordpairs): #return []
        num_shuffle=4
        sentence_listids = sentence_to_listofids(sentence)
        sentence_token_list = tokenizer.convert_ids_to_tokens(sentence_listids)
        instances = [] #list of (sent_ids,label)
        for word1,word2 in wordpairs:
            positive_sentence_listids = copy.deepcopy(sentence_listids)
            negative_sentence_listids = copy.deepcopy(sentence_listids)
            word1 = str(word1,encoding='utf-8')
            word2 = str(word2,encoding='utf-8')
            word1_ind = tokenizer(word1)['input_ids'][1]#.squeeze().item() #int
            word2_ind = tokenizer(word2)['input_ids'][1]
            
            positive_sentence_listids[hole1_ind] = word1_ind
            positive_sentence_listids[hole2_ind] = word2_ind
            instances.append((positive_sentence_listids,1))

            negative_sentence_listids[hole1_ind] = word2_ind
            negative_sentence_listids[hole2_ind] = word1_ind
            instances.append((negative_sentence_listids,0))
        
        random_pairs = random.sample(wordpairs,num_shuffle)
        for i in range(num_shuffle):
            if i == num_shuffle-1:
                wrong_pair = (random_pairs[i][0],random_pairs[0][1])
                negative_sentence_listids = copy.deepcopy(sentence_listids)
                word1 = str(wrong_pair[0],encoding='utf-8')
                word2 = str(wrong_pair[1],encoding='utf-8')
                word1_ind = tokenizer(word1)['input_ids'][1]#.squeeze().item() #int
                word2_ind = tokenizer(word2)['input_ids'][1]

                negative_sentence_listids[hole1_ind] = word1_ind
                negative_sentence_listids[hole2_ind] = word2_ind
                instances.append((negative_sentence_listids,0))
                
            else:
                wrong_pair = (random_pairs[i][0],random_pairs[i+1][1])
                negative_sentence_listids = copy.deepcopy(sentence_listids)
                word1 = str(wrong_pair[0],encoding='utf-8')
                word2 = str(wrong_pair[1],encoding='utf-8')
                word1_ind = tokenizer(word1)['input_ids'][1]#.squeeze().item() #int
                word2_ind = tokenizer(word2)['input_ids'][1]

                negative_sentence_listids[hole1_ind] = word1_ind
                negative_sentence_listids[hole2_ind] = word2_ind
                instances.append((negative_sentence_listids,0))
        
        return instances



def get_init_transformer(transformer):

  def init_transformer(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=transformer.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

  return init_transformer



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.encoder = BertModel.from_pretrained('C:/Alchemy/bert_model/bert-base-uncased/')
        self.linear = nn.Linear(768,2)
        #self.linear.apply(get_init_transformer(self.encoder))
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
    def forward(self,input_ids,label,attention_mask):#([batchsize,len],[batchsize])
        input_ids = input_ids.int()#torch.LongTensor(input_ids.to('cpu'))#.to('cuda')
        label = label.squeeze()
        #label = torch.LongTensor(label)
        bertoutput =  self.encoder(input_ids,attention_mask=attention_mask)#,attention_mask=word_mask.to('cuda'))
        CLSembedding = bertoutput[1]
        #print("CLSemb shape",CLSembedding.shape)
        logits = self.linear(CLSembedding)
        #print('logits shape:',logits.shape)
        loss = self.loss(logits,label)
        return loss,logits
    def predict(self,input_ids,attention_mask=None):
        input_ids = input_ids.int()#torch.LongTensor(input_ids.to('cpu'))#.to('cuda')
        #label = torch.LongTensor(label)
        bertoutput =  self.encoder(input_ids,attention_mask=attention_mask)#,attention_mask=word_mask.to('cuda'))
        CLSembedding = bertoutput[1]
        #print("CLSemb shape",CLSembedding.shape)
        logits = self.linear(CLSembedding)
        #print('logits shape:',logits.shape)
        return logits



def train(model, dataloader_train, optimizer, num_epochs=10, clip=0., verbose=True, device='cuda', select_model=False):
    model = model.to(device)  
    loss_avg = float('inf')
    acc_train = 0.
    best_acc_val = 0.
    start_time = datetime.now() 
    best_state_dict = None
    num_continuous_fails = 0
    tolerance = 6

    count = 0
    for epoch in range(num_epochs):
        model.train()  
        loss_total = 0.
        num_correct_train = 0
        count+=1
        print('epoch count:',count)
        for batch_ind, batch in enumerate(dataloader_train):
            sentids, labels, masks = batch
            sentids = sentids.to(device) 
            labels = labels.to(device)
            masks = masks.to(device)
            loss_batch_total,logits = model(sentids,labels,masks)  
            preds = torch.where(logits > 0., 1, 0)  
            num_correct_train += (preds == labels).sum()
            loss_total += loss_batch_total.item()            
        
        if math.isnan(loss_total):  
            break
        
        loss_batch_avg = loss_batch_total / sentids.size(0)  
        loss_batch_avg.backward()  

        if clip > 0.:  
            nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step() 
        optimizer.zero_grad()  

        if math.isnan(loss_total):
            print('Stopping training because loss is NaN')
            break

        
        loss_avg = loss_total / len(dataloader_train.dataset)
        acc_train = num_correct_train / len(dataloader_train.dataset) * 100.

      
        print('Epoch {:3d} | avg loss {:8.4f}| train acc {:2.2f} '.format(epoch + 1, loss_avg,acc_train))
    model.eval()
    return model

def predict_prob(model,wordpair,templates):#(model,(b'w1,b'w2),(sentence,hole1,hole2,score))
    correct_count = 0
    count=0
    word1,word2 = wordpair
    word1 = str(word1,encoding='utf-8')
    word2 = str(word2,encoding='utf-8')
    word1_ind = tokenizer(word1)['input_ids'][1]#.squeeze().item() #int
    word2_ind = tokenizer(word2)['input_ids'][1]
    count_porb = 0
    for template in templates:
        count+=1
        sentence,hole1,hole2,score = template
        sentence_listids = sentence_to_listofids(sentence)
        sentence_listids[int(hole1)] = word1_ind
        sentence_listids[int(hole2)] = word2_ind
        senteceids_tensor = torch.tensor([sentence_listids])
        logit = model.predict(senteceids_tensor)
        prob = F.softmax(logit,dim=1)
        pred = torch.argmax(prob,dim=1).squeeze().item()
        #if pred == 1:
        #    correct_count+=1
        #print(prob)
        prob_positive = prob.squeeze()[1].item()
        print(prob_positive)
        count_porb+=prob_positive
        #print(pred)
        ###
    #return correct_count/count>0.46
    print(count_porb/count)
    return (count_porb/count)>0.5

def predict_max(model,wordpair,templates):#(model,(b'w1,b'w2),(sentence,hole1,hole2,score))
    word1,word2 = wordpair
    word1 = str(word1,encoding='utf-8')
    word2 = str(word2,encoding='utf-8')
    word1_ind = tokenizer(word1)['input_ids'][1]#.squeeze().item() #int
    word2_ind = tokenizer(word2)['input_ids'][1]

    max_prob_pos = 0
    min_prob_pos = 1
    for template in templates:

        sentence,hole1,hole2,score = template
        sentence_listids = sentence_to_listofids(sentence)
        sentence_listids[int(hole1)] = word1_ind
        sentence_listids[int(hole2)] = word2_ind
        senteceids_tensor = torch.tensor([sentence_listids])
        logit = model.predict(senteceids_tensor)
        prob_pos = F.softmax(logit,dim=1).squeeze()[1].item()
        prob_neg = 1-prob_pos

        if prob_pos>max_prob_pos:
            max_prob_pos = prob_pos
        if prob_pos<min_prob_pos:
            min_prob_pos = prob_pos
        #pred = torch.argmax(prob,dim=1).squeeze().item()
        #if pred == 1:
        #    correct_count+=1
        #print(prob)
        #print(prob.squeeze()[1].item())
        #print(pred)
        ###
    print(max_prob_pos-(1-min_prob_pos))
    if max_prob_pos>(1-min_prob_pos):
        return True
    else:
        return False


'''
def testpredict():
    model = Classifier()
    dataset = MyDataSet(scoretemp_male_female_path,wordpairpath_male_female,k=10)
    templates = dataset.templates
    wordpairs = GetPairs(wordpairpath_male_female)
    count = 0
    for wordpair in wordpairs:
        pred = predict(model,wordpair,templates)
        if pred:
            count+=1
        #print(pred)
    return count/len(wordpairs)
'''

def predict_prob_countacc(model,wordpairs,templates):
    count = 0
    l = len(wordpairs)
    for wordpair in wordpairs:
        word1,word2 = wordpair
        negative_pair = (word2,word1)    
        
        pred_positive = predict_prob(model,wordpair,templates)
        if pred_positive==True:
            count+=1
        #print(pred)
        
        pred_negative = predict_prob(model,negative_pair,templates)
        if pred_negative==False:
            count+=1
        
    #for i in range(l):

    return count/(len(wordpairs)*2)

def predict_max_countacc(model,wordpairs,templates):
    count = 0
    for wordpair in wordpairs:
        word1,word2 = wordpair
        negative_pair = (word2,word1)    
        
        pred_positive = predict_max(model,wordpair,templates)
        if pred_positive==True:
            count+=1
        #print(pred)

        pred_negative = predict_max(model,negative_pair,templates)
        if pred_negative==False:
            count+=1

    return count/(len(wordpairs)*2)

def testtrain():
    model = Classifier()
    dataset = MyDataSet(scoretemp_animal_shelter,wordpairpath_animal_shelter,k=10)
    val_wordpairs = GetPairs(wordpairpath_animal_shelter)
    templates = dataset.templates
    '''
    with open('datasetdump_male_female_top10.txt', 'wb') as tempfile:
        pickle.dump(dataset, tempfile)
    '''
    print(len(dataset))


    dataloader = DataLoader(dataset,2,shuffle=True)


    trained_model = train(model, dataloader, torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0003), clip=0., num_epochs=5, verbose=True, device='cuda')
    #trained_model = model
    
    val_wordpairs.reverse()
    len_dataset = len(val_wordpairs)
    val_wordpairs = val_wordpairs[0:int(len_dataset*(1-split_rate))]
    #acc = predict_prob_countacc(trained_model.to('cpu'),val_wordpairs,templates)#######
    acc = predict_max_countacc(trained_model.to('cpu'),val_wordpairs,templates)#######
    print('final acc:',acc)

if __name__ == "__main__":
    testtrain()
    