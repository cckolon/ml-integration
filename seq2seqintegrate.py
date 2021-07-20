# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:18:12 2021

Source for symbolic integration w/ neural networks: https://arxiv.org/pdf/1912.01412.pdf
Source for Seq2Seq how to: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
Source for trees in python: https://medium.com/swlh/making-data-trees-in-python-3a3ceb050cfd
Source for RPN comprehension: https://blog.klipse.tech/python/2016/09/22/python-reverse-polish-evaluator.html
Source for jit: http://numba.pydata.org/

@author: Carl Kolon
"""
import random as rand
import numpy as np
import sympy as sp
import itertools as itt
from numba import jit
from io import open
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import json
import time
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} for computation.")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 40

unaryoperators = ["log","exp","sin","cos","sqrt","asin","acos","atan"]
binaryoperators = ['+','-','*','/','**']
variables = ['x','1','2','3']

class Lang:
	def __init__(self):
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2
	
	def addphrase(self,phrase):
		for word in phrase:
			self.addword(word)
	
	def addword(self,word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

def loadtrainingdata(filepath):
	pairs = []
	with open(filepath) as file:
		for line in file.readlines():
			pair = json.loads(line)
			if len(pair[0]) < MAX_LENGTH:
				if len(pair[1]) < MAX_LENGTH:
					pairs.append(pair)
	print(f"I imported {len(pairs)} pairs.")
	return pairs

def prepareData(inputlang,outputlang,filepath):
	pairs = loadtrainingdata(filepath)
	for pair in pairs:
		inputlang.addphrase(pair[0])
		outputlang.addphrase(pair[1])
	return pairs

integrands = Lang()
integrals = Lang()

pairs = prepareData(integrands,integrals,'trainingdata.txt')

class EncoderRNN(nn.Module):
	def __init__(self,input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(input_size,hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
	
	def forward(self, inseq, hidden):
		embedded = self.embedding(inseq).view(1,1,-1)
		output = embedded
		output, hidden = self.gru(output,hidden)
		return output, hidden
	
	def initHidden(self):
		return torch.zeros(1,1,self.hidden_size,device=device)

class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(output_size,hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size,output_size)
		self.softmax = nn.LogSoftmax(dim=1)
	
	def forward(self, inseq, hidden):
		output = self.embedding(inseq).view(1,1,-1)
		output = F.relu(output)
		output, hidden = self.gru(output,hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden
	
	def initHidden(self):
		return torch.zeros(1,1,self.hidden_size, device=device)
	
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inseq, hidden, encoder_outputs):
        embedded = self.embedding(inseq).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
	
def indicesfromphrase(lang, phrase):
	return [lang.word2index[word] for word in phrase]

def tensorfromphrase(lang, phrase):
	indices = indicesfromphrase(lang, phrase)
	indices.append(EOS_token)
	return torch.tensor(indices, dtype = torch.long, device=device).view(-1,1)

def tensorsfrompair(input_lang,output_lang,pair):
	input_tensor = tensorfromphrase(input_lang, pair[0])
	target_tensor= tensorfromphrase(output_lang, pair[1])
	return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.8


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if rand.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = np.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
	plt.plot(points)
	#fig = plt.pyplot.subplot()
	#ax = fig.axis()
	#loc = ticker.MultipleLocator(base=0.2)
	#ax.yaxis.set_major_locator(loc)
	#plt.plot(points)

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsfrompair(integrands,integrals,rand.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
	
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorfromphrase(integrands, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(integrals.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = rand.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        print('<', output_words)
        print('')
		
hidden_size = 256
encoder1 = EncoderRNN(integrands.n_words, hidden_size).to(device)

try:
	if device.type == 'cpu':
		encoder1.load_state_dict(torch.load('encoder1',map_location=torch.device('cpu')))
	else:
		encoder1.load_state_dict(torch.load('encoder1'))
	print('Loaded encoder from file.')
except:
	print('No encoder found. Generating randomly.')

attn_decoder1 = AttnDecoderRNN(hidden_size, integrals.n_words, dropout_p=0.1).to(device)

try:
	if device.type == 'cpu':
		attn_decoder1.load_state_dict(torch.load('attn_decoder1',map_location=torch.device('cpu')))
	else:
		attn_decoder1.load_state_dict(torch.load('attn_decoder1'))
	print('Loaded decoder from file.')
except:
	print('No decoder found. Generating randomly.')

try:
	trainIters(encoder1, attn_decoder1, 100000, print_every=500,learning_rate = 0.0001)
finally:
	evaluateRandomly(encoder1, attn_decoder1)
	torch.save(encoder1.state_dict(),'encoder1')
	torch.save(attn_decoder1.state_dict(),'attn_decoder1')