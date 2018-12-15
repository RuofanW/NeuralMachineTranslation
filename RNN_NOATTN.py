from __future__ import unicode_literals, print_function, division
import pickle as pkl
from io import open
import unicodedata
import string
import re
import random
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np, pandas as pd


import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
teacher_forcing_ratio = 0
import math


from sacrebleu import raw_corpus_bleu, corpus_bleu

print('RNN no ATTN, no pre trained ebd, no teacher forcing, lr=0.001,hid=512,bs=64, no sort shuff, maxlen= 38, eval every =100')

def mask_ind(arr):
    arr = arr.cpu().numpy()
    batch_size = arr.shape[1]

    for i in range(batch_size):
        if 1 in arr[:,i]:
            ind = np.where(arr[:,i]== 1)[0][0]
        
            arr[:,i][:ind+1]=1
            arr[:,i][ind+1:]=0
        else:
            arr[:,i]=1
        
    
    return arr, np.count_nonzero(arr)
                

def convert_idx_2_sent_new(idx_tensor, lang_obj):
    word_list = []
    #truth_word_list = []
    for i in idx_tensor:
        if i.item() not in set([PAD_IDX,EOS_token,SOS_token]):
            word_list.append(lang_obj.index2word[i.item()])
#     for j in truth_tensor:
#         if j.item() not in set([PAD_IDX,EOS_token,SOS_token]):
#             truth_word_list.append(lang_obj.index2word[j.item()])
    sent = (' ').join(word_list)
    #truth_sent = (' ').join(truth_word_list)
    return sent


def bleu_new(corpus,truths):
    n = len(corpus)
    #bleu = [0]*n
    pred_ls = []
    true_ls = []
    for i in range(n):
        pred, true = corpus[i], truths[i]
        pred_ls.append( [convert_idx_2_sent_new(sent, target_tra) for sent in pred])
        true_ls.append([convert_idx_2_sent_new(sent, target_tra) for sent in true])
    flattened_pred  = [val for sublist in pred_ls for val in sublist]
    flattened_true  = [val for sublist in true_ls for val in sublist]
    bleu= corpus_bleu(flattened_pred, [flattened_true]).score
    return bleu
    
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'


SOS_token = 0
EOS_token = 1
PAD_IDX = 2
UNK_IDX = 3
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD", 3:"UNK"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
            
def normalizeString(s):
#     s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"&apos;m", r"am", s)
    s = re.sub(r"&apos;s", r"is", s)
    s = re.sub(r"&apos;re", r"are", s)
    s = re.sub(r"&apos;", r"", s)
    return s
    

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def loadingLangs(sourcelang, targetlang, setname):
    input_ls = []
    output_ls = []
    print('Reading lines...')
    # Read the file 
    with open('../iwslt-%s-%s/%s.tok.%s'%(sourcelang, targetlang, setname,sourcelang)) as f:
        for line in f.readlines():
            input_ls.append([normalizeString(word) for word in line.split()])
    with open('../iwslt-%s-%s/%s.tok.%s'%(sourcelang, targetlang, setname,targetlang)) as f:
        for line in f.readlines():
            output_ls.append([normalizeString(word) for word in line.split()])
    pairs = list(zip(input_ls, output_ls))
    print('Read %s sentence pairs'%(len(input_ls)))
    input_lang = Lang(sourcelang)
    output_lang = Lang(targetlang)
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
    
    
source_tra, target_tra, pairs_tra = loadingLangs('zh', 'en', 'train')
source_val, target_val, pairs_val = loadingLangs('zh', 'en', 'dev')
source_tes, target_tes, pairs_tes = loadingLangs('zh', 'en', 'test')

MAX_SENT_LEN = 38
BATCH_SIZE = 64

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else UNK_IDX for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair,source,target):
    input_lang = source
    output_lang = target
    input_tensor = tensorFromSentence(input_lang, pair[0]).reshape((-1))
    target_tensor = tensorFromSentence(output_lang, pair[1]).reshape((-1))
    return (input_tensor, input_tensor.shape[0], target_tensor, target_tensor.shape[0])


class NMTDataset(Dataset):
    def __init__(self, source, target, pairs):
        self.source = source
        self.target = target
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        inp_ten, inp_len, tar_ten, tar_len = tensorsFromPair(self.pairs[key], self.source, self.target)
        item = {}
        item['inputtensor'] = inp_ten[:MAX_SENT_LEN]
        item['inputlen'] = min(inp_len, MAX_SENT_LEN)
        item['targettensor'] = tar_ten[:MAX_SENT_LEN]
        item['targetlen'] = min(tar_len, MAX_SENT_LEN)
        return item
        
train_data = NMTDataset(source_tra, target_tra, pairs_tra)
val_data = NMTDataset(source_tra, target_tra, pairs_val)
test_data = NMTDataset(source_tra, target_tra, pairs_tes)


#collate function

def collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    src_data, tar_data, src_len, tar_len = [], [], [], []
    for datum in batch:        
        src_datum = np.pad(np.array(datum['inputtensor']),
                                pad_width=((0,MAX_SENT_LEN-datum['inputlen'])),
                                mode="constant", constant_values=PAD_IDX)
        tar_datum = np.pad(np.array(datum['targettensor']),
                                pad_width=((0,MAX_SENT_LEN-datum['targetlen'])),
                                mode="constant", constant_values=PAD_IDX)
        src_data.append(src_datum)
        tar_data.append(tar_datum)
        src_len.append(datum['inputlen'])
        tar_len.append(datum['targetlen'])
    return [torch.from_numpy(np.array(src_data)).to(device),torch.from_numpy(np.array(tar_data)).to(device),
               torch.from_numpy(np.array(src_len)).to(device),torch.from_numpy(np.array(tar_len)).to(device)]
               
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_func)

#Here change to shuffle=True
val_loader = torch.utils.data.DataLoader(val_data,
                                           batch_size=BATCH_SIZE,shuffle=True, collate_fn=collate_func)


# sample data loader
count = 0
for data in train_loader:
    count+=1
    print('input sentence tensor batch: ')
    print(data[0][3])
    print('input sentence token batch: ')
    
    print('input batch dimension: {}'.format(data[0].size()))
    print('target sentence batch: ')
    print(data[1][3])
    tokens = convert_idx_2_sent_new(data[1][3], target_tra)
    print(tokens)
    
    print('target batch dimension: {}'.format(data[1].size()))
    print('input sentence len: ')
    print(data[2])
    print('target sentence len: ')
    print(data[3])
    if count == 1:
        break
        
        
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx =PAD_IDX)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        
    def forward(self, input, hidden):
        batch_size = input.size()[1]
        #print('in Encoder, batch_size is {} \n'.format(batch_size))
        seq_len = input.size()[0]
        #print('in Encoder, seq_len is {} \n'.format(seq_len))
        embedded = self.embedding(input).view(seq_len, batch_size, -1)
        #print('in Encoder, embedded is {}, dimension is {} \n'.format(embedded, embedded.size()))
        output = embedded
        
        output, hidden = self.gru(output, hidden)
        #print('in Encoder, output after gru is {}, dimension is {} \n'.format(output, output.size()))
        #print('in Encoder, hidden after gru is {}, dimension is {} \n'.format(hidden, hidden.size()))
        
        context = self.fc1(torch.cat((hidden[0],hidden[1]),dim = 1 )).unsqueeze(0)
        #print('in encoder, context is {}, dimension is {}'.format(context, context.size()))
        #output = self.fc1(output)
        return output, context

    def initHidden(self, batch_size):
        initHidden = torch.zeros(2, batch_size, self.hidden_size, device=device)
        #print('in Encoder, initialized hidden dimension is {}'.format(initHidden.size()))
        return initHidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx = PAD_IDX)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1,-1)
        batch_size = input.size()[1]
        #print('in Decoder, batch_size is {}'.format(batch_size))
        
        #print('in Decoder, input before embedded layer is {}, dimension is {}'.format(input,input.size()))
        output = self.embedding(input).view(1, batch_size, -1)
        #print('in Decoder, output after embedded is {}, dimension is {} \n'.format(output, output.size()))
        output = F.relu(output)
        #print('in Decoder, output after relu is {}, dimension is {} \n'.format(output, output.size()))
        #print('in Decoder, the initial hidden is {}, dimension is {}'.format(hidden, hidden.size()))
        
        output, hidden = self.gru(output, hidden)
        #print('in Decoder, output of GRU is {}, dimension is {}'.format(output, output.size()))
        #print('in Decoder, hidden of GRU is {}, dimension is {}'.format(hidden, hidden.size()))
        
        output = self.softmax(self.out(output[0]))
        #print('in Decoder, output after softmax is {}, dimension is {}'.format(output, output.size()))
        return output, hidden

    def initHidden(self, batch_size):
        initHidden = torch.zeros(1, batch_size, self.hidden_size, device=device)
        #print('in Decoder, initHidden is {}, dimension is {} \n'.format(initHidden, initHidden.size()))
        return initHidden
        
        
    
def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer):
    
    batch_size = input_tensor.size()[1]
    #print('in train, batch size is {}'.format(batch_size))
    encoder_hidden = encoder.initHidden(batch_size)
    #print('in train, initial encoder hidden is {}, dimension is {}'.format(encoder_hidden, encoder_hidden.size()))
    
    encoder_optimizer.zero_grad()  
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size()[0]
    #print('in train, input_length is {}'.format(input_length))
    target_length = target_tensor.size()[0]
    #print('in train, target_length is {}'.format(target_length))
    #encoder_outputs = torch.zeros(target_length, batch_size, encoder.hidden_size, device=device) 

  

    _, context = encoder(input_tensor, encoder_hidden)
    #print('in train encoder_hidden[0] is {}, dimension is {}'.format(encoder_hidden[0],encoder_hidden[0].size()))
    #print('in train after concatenating encoder_hidden[0] and [1] is {}, dimension is {}'.format(torch.cat((encoder_hidden[0].cpu().data,encoder_hidden[1].cpu().data),dim = 1), torch.cat((encoder_hidden[0].cpu().data,encoder_hidden[1].cpu().data),dim = 1).size()))
    
    #decoder_hidden = nn.Linear(2*hidden_size,hidden_size)(
        #torch.cat((encoder_hidden[0].cpu().data,encoder_hidden[1].cpu().data),dim = 1)).to(device).unsqueeze(0)
    
    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device)  # decoder_input: torch.Size([1, 32])
    decoder_hidden = context.to(device)
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #print('target_tensor is {}, dimension is {}'.format(target_tensor, target_tensor.size()))
    
    
    #print('target_tensor is {}, dimension is {}'.format(target_tensor, target_tensor.size()))
    #print('sentence 3 in this batch is {}, dimension is {}'.format(convert_idx_2_sent_new(target_tensor[:,2], target_tra)))
    #print('sentence 3 in this batch is {}'.format(convert_idx_2_sent_new(target_tensor[:,2], target_tra)))
    
    if use_teacher_forcing:
        loss = 0 
        criterion = nn.NLLLoss(reduce = True, ignore_index = 2, reduction = 'mean') 

        for di in range(target_length):
            #print('in teacher_forcing, step {}'.format(di))
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            
            decoder_input = target_tensor[di]  
            #print('in teacher forcing, decoder_output at current timestep is {}, dimension is {}'.format(decoder_output, decoder_output.size()))
            #print('predicted target at current timestep is {}, dimension is {}'.format(torch.argmax(decoder_output, dim=1), torch.argmax(decoder_output, dim=1).size()))
            #print('true target at current timestep is {}, dimension is {}'.format(target_tensor[i], target_tensor[i].size()))
            #print('predicted target at current timestep is {}, dimension is {}'.format(decoder_output, decoder_output.size()))
            
            temp_loss = criterion(decoder_output, target_tensor[di])
            #print ('in teacher forcing, temp loss at current step is {}'.format(temp_loss))
            #print('temp_loss for current batch, current token is {}, dimension is {}'.format(temp_loss, temp_loss.size()))
            
            loss += temp_loss
            #loss += temp_loss * mask[di:di+1].float()  
            #print('loss is {}, dimension is {}'.format(loss, loss.size()))
            #ave_loss = loss.sum()/batch_size
        ave_loss = loss/target_length
            
    else:
        loss = None 
        criterion = nn.NLLLoss(reduce = False) 
        prediction = None

        for di in range(target_length):
            #print('in non_teacher forcing, step {}'.format(di))
            
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            #print('in non_teacher forcing, topi is {}, dimension is {}'.format(topi, topi.size()))
            
            if prediction is None:
                prediction = topi.view(1,-1)
            else:
                prediction = torch.cat((prediction, topi.view(1,-1)), dim=0)
            
            #print('at current step, cumulative prediction is {}, dimension is {}'.format(prediction, prediction.size()))
            
                            
            decoder_input = topi.transpose(0,1).detach()  # detach from history as input
            #print('in non_teacher forcing, input of the current step is {}, dimension is {}'.format(topi.transpose(0,1),topi.transpose(0,1).size()))
            #print('in non_teacher forcing decoder_output at current timestep is {}, dimension is {}'.format(decoder_output, decoder_output.size()))
            
            #print('predicted target at current timestep is {}, dimension is {}'.format(torch.argmax(decoder_output, dim=1), torch.argmax(decoder_output, dim=1).size()))

            #print('true target at current timestep is {}, dimension is {}'.format(target_tensor[i], target_tensor[i].size()))
            
            temp_loss = criterion(decoder_output, target_tensor[di])
            if loss is None:
                loss = temp_loss.view(1,-1)
            else:
                loss = torch.cat((loss, temp_loss.view(1,-1)),dim=0)
            #print('temp_loss for current batch, current token is {}, dimension is {}'.format(temp_loss, temp_loss.size()))
            
            
    
    #print('Final prediction is {}'.format(prediction))
        mask, count = mask_ind(prediction)
        total_loss = torch.sum(loss * torch.from_numpy(mask).float().to(device))
        ave_loss = total_loss/count
    #print('total_loss is {}, dimension is{}'.format(total_loss, total_loss.size()))        
    ave_loss.backward()
    
    
    encoder_optimizer.step()   
    decoder_optimizer.step()
    
    #print('total valid predicted token is {}'.format(count))
    #print('ave_loss type is {}'.format(type(ave_loss)))
    #print('ave_loss.item() type is {}'.format(type(ave_loss.item())))
    
    return ave_loss.item()


def evaluate(encoder, decoder, data_loader, max_length=MAX_SENT_LEN):
    start = time.time()
    encoder.eval()
    decoder.eval()
    inputs = []
    corpus = []
    truths = []
    for i, (input_sentences, target_sentences,len1,len2) in enumerate(data_loader):
#         if i % 5 == 0:
#             print('Time: {}, Step: [{}/{}]'.format(
#                 timeSince(start, i + 1/len(train_loader)), i, len(data_loader)))
        inputs.append(input_sentences.to(device))#put into inputs: batch*seq: each row is a sentence
        input_tensor = input_sentences.transpose(0,1).to(device)
        truths.append(target_sentences.to(device))#put into truths: batch*seq: each row is a sentence
        target_tensor = target_sentences.transpose(0,1).to(device) 
        #truths.append(target_tensor)
        input_length = input_tensor.size()[0]
        batch_size = input_tensor.size()[1]
    
        
        encoder_hidden = encoder.initHidden(batch_size)
        #encoder_outputs = torch.zeros(max_length, batch_size, encoder.hidden_size, device=device)
        _, context = encoder(input_tensor, encoder_hidden)
        
        #encoder_hidden = nn.Linear(2*hidden_size,hidden_size)(
        #torch.cat((encoder_hidden[0].cpu().data,encoder_hidden[1].cpu().data),dim = 1)).to(device).unsqueeze(0)
        
        decoder_hidden = context.to(device)
        decoder_input = torch.tensor([[SOS_token]*batch_size], device=device) 
        decoded_words = torch.zeros(batch_size, max_length)
    
    

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoded_words[:,di] = topi.squeeze()  #put into decoded_words: batch*seq
            decoder_input = topi.transpose(0,1).detach()
            #print('true target is {}, dimension is {}'.format(target_tensor[:,di],target_tensor[di].size()))
            #print('before transpose, topi is {}, dimension is {}'.format(topi, topi.size()))
            #print('after transpose, topi is {}, dimension is {}'.format(topi.transpose(0,1),topi.transpose(0,1).size()))
        corpus.append(decoded_words)
        
        #print('last: decoded_words is {}, dimension is {}'.format(decoded_words, decoded_words.size()))
        #print('last: inputs is {}, dimension is {}'.format(inputs, len(inputs)))
        #print('last: truths is {}, dimension is {}'.format(truths, len(truths)))
        #print(inputs[0].size(), corpus[0].size(), truths[0].size())
    return inputs, corpus, truths




hidden_size = 512
learning_rate=0.001
num_epoch = 30
print_every = 100
plot_every = 100

encoder1 = EncoderRNN(source_tra.n_words,hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, target_tra.n_words).to(device)

start = time.time()

encoder_optimizer = optim.Adam(encoder1.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder1.parameters(), lr=learning_rate)


whole_train_loss = []
whole_val_bleu = []
for epoch in range(1, num_epoch + 1):
    plot_bleu_score_val = []

    plot_losses = []
    print_loss_total = 0  
    plot_loss_total = 0  
    for i, (input_sentences, target_sentences,len1,len2) in enumerate(train_loader): 
#         print(i)
        encoder1.train()
        decoder1.train()
        input_tensor = input_sentences.transpose(0,1)   
        target_tensor = target_sentences.transpose(0,1)
        loss = train(input_tensor, target_tensor, encoder1,
                     decoder1, encoder_optimizer, decoder_optimizer)
        print_loss_total += loss
        plot_loss_total += loss
        
        if i > 0 and i % print_every == 0:
            inputs, corpus, truths = evaluate(encoder1, decoder1, val_loader, max_length=MAX_SENT_LEN)
            bleu_score_val_avg = bleu_new(corpus, truths)#np.mean(bleu_score_val)

            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Time: {}, Epoch: [{}/{}], Step: [{}/{}], Train Loss: {}, BLEU: {}'.format(
                timeSince(start, i + 1/len(train_loader)), epoch, num_epoch, i, 
                len(train_loader),print_loss_avg,bleu_score_val_avg))
            
            plot_bleu_score_val.append(bleu_score_val_avg)


            
            print('\nInput1:> %s'%(' '.join([source_tra.index2word[i.item()] for i in inputs[0][3] if i.item() not in set([PAD_IDX,EOS_token,SOS_token])])))
            print('\nTarget1:= %s'%(convert_idx_2_sent_new(truths[0][3], target_tra)),
                    '\nPredict1:< %s' %(convert_idx_2_sent_new(corpus[0][3], target_tra)))
            
            print('\nInput2:> %s'%(' '.join([source_tra.index2word[i.item()] for i in inputs[1][3] if i.item() not in set([PAD_IDX,EOS_token,SOS_token])])))
            print('\nTarget2:= %s'%(convert_idx_2_sent_new(truths[1][3], target_tra)),
                    '\nPredict2:< %s' %(convert_idx_2_sent_new(corpus[1][3], target_tra)))

        if i > 0 and i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
    print(plot_losses)
    whole_train_loss.append(plot_losses)
    whole_val_bleu.append(plot_bleu_score_val)



storedir = '/scratch/zh1087/NMT/'
pkl.dump(whole_val_bleu, open(storedir+'NOATTNRNN_CONFIG1_bleu_score_list.pkl','wb'))
pkl.dump(whole_train_loss,open(storedir+'NOATTNRNN_CONFIG1_train_loss.pkl','wb'))
pkl.dump(truths, open(storedir+'NOATTNRNN_CONFIG1_truths.pkl','wb'))
pkl.dump(corpus,open(storedir+'NOATTNRNN_CONFIG1_corpus.pkl','wb'))

state ={'state_dict_enc':encoder1.state_dict(),'state_dict_dec':decoder1.state_dict()}
torch.save(state,storedir+'NOATTNRNN_CONFIG1_modles.pkl')
       
                
