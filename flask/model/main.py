import numpy as np
import re

from bs4 import BeautifulSoup

import spacy
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

from gensim.models import KeyedVectors


import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, bidirectional):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if bidirectional: self.bidirectional = 2
        else: self.bidirectional = 1
        
        if num_layers < 2:
            dropout = 0
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size*self.bidirectional, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden, c):
        output, (hidden, c) = self.lstm(input, (hidden, c))
        output = self.dropout(output)
        output = output.squeeze()[-1].unsqueeze(0)
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden, c
    
    def initHidden(self):
        return torch.zeros(self.bidirectional*self.num_layers, 1, self.hidden_size)
    def initC(self):
        return torch.zeros(self.bidirectional*self.num_layers, 1, self.hidden_size)


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def lemma(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc 
          if (token.lemma_.isalnum() or token.lemma_ in ['.', '!', '?'] ) and token.lemma_ not in stopwords]
    text = ' '.join(lemmas)
    return text

def clean(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = lemma(text)
    return text

def getLabel(output):
    value_tensor, index_tensor = output.topk(1)
    return index_tensor[0].item()

def test(lstm, X_test, y_test):
    n = len(X_test)
    correct = 0
    y_pred = []
    
    for i in range(n):
        with torch.no_grad():
            hidden, c = lstm.initHidden(), lstm.initC()
            output, hidden, c = lstm(torch.tensor(X_test[i]).unsqueeze(1), hidden, c)
            
            y_pred.append(getLabel(output))

            if y_pred[-1] == y_test[i]:
                correct+=1
    
    if y_pred[-1] == 1:
        pred_sentiment = 'Positive'
    else:
        pred_sentiment = 'Negative'
    
    return pred_sentiment


def getPrediction(input):

    lstm = LSTM(100,128,2,1,.1,True)
    lstm.load_state_dict(torch.load('./model/lstm.pth'))
    lstm.eval()

    word2vec_model = KeyedVectors.load('./model/word2vec_model', mmap='r')

    temp = []
    cleaned_text = clean(input)
    for word in cleaned_text.split():
        try:
            temp.append(list(word2vec_model.wv.get_vector(word)))
        except:
            pass
    
    pred_sentiment = test(lstm, np.array([temp]), [0])

    return cleaned_text, pred_sentiment