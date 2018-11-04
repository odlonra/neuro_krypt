# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:16:49 2018

@author:    Arne Goerlitz
@date:      10/27/2018
"""
# import os
# import sys
# import pathlib as pl
# import bs4 as soup

import torch
import torch.nn as nn
import torch.optim as optim
#from torch.autograd import Variable


###############################################################################
###############################################################################
class Krypt(nn.Module):
    def __init__(self):
        super(Krypt, self).__init__()

        self.lin1 = nn.Linear(128,500)
        self.lin2 = nn.Linear(500,1000)
        self.lin3 = nn.Linear(1000,128)

    def forward(self, x):
        x = torch.relu_(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


###############################################################################
class DeKrypt(nn.Module):
    def __init__(self):
        super(DeKrypt, self).__init__()
        self.lin1 = nn.Linear(128, 100)
        self.lin2 = nn.Linear(100, 500)
        self.lin3 = nn.Linear(500, 128)

    def forward(self, x):
        x = torch.relu_(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


###############################################################################
###############################################################################
'''
This function translates a string into an tensor.
'''
def word2tensor(word):
    tensor = torch.zeros(len(word),128)
#    word += (10-len(word))*' '  
    i = 0
    for char in word:   
        tensor[i][ord(char)] = 1
        i= i+1
    i = 0
    return tensor
'''
This function translates a tensor back into a string.
'''    
def tensor2word(tensor):
    word = ""
    i=0
    for column in tensor:
        index = column.argmax()

        word += chr(index)
        i=i+1
    return word

def main():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    message = 'Arne ist im Zug.'
    m_tens = word2tensor(message)
    m_tens = m_tens.cuda()
 #   word = tensor2word(tens)
 #   print(word)

#    krypt = Krypt()
#    dekrypt = Krypt()
#    dekrypt.cuda()
#    spy = Spy()
    
    test = Krypt()
    test = test.cuda()

    input = m_tens
    for i in range(2000):
#        print("\n\nMessage Input:\n")
#        print(input)

        #output = krypt(input)
#        print(output)


#        message_krypt = test(input)
        
 #       print('Kryp: '+tensor2word(message_krypt))
        
        
#        message_out = dekrypt(message_krypt)
        message_out =  test(input)
       
        #message_out = dekrypt(output)
#        print("\n\nMessage Output:\n")
#        print(message_out)


        #bugging = spy(output)
#        print("\n\nBugging:\n")
#        print(bugging)

        ###Test
#        targ_tens = torch.zeros(10, 128)
#        targ_tens[0][0] = 1
#        targ_tens[1][17] = 1
#        targ_tens[2][13] = 1
#        targ_tens[3][4] = 1
#        target = Variable(targ_tens)
#        print(target)
        ###

#        crit_krypt = nn.MSELoss()
#        loss_krypt = crit_krypt(target, message_out)
#        krypt.zero_grad()
#        loss_krypt.backward()
#        opti_krypt = optim.SGD(krypt.parameters(), lr=0.5)
#        opti_krypt.step()
        
        crit_test = nn.MSELoss()
        loss_test = crit_test(message_out, m_tens)
        test.zero_grad()
        loss_test.backward()
        opti_test = optim.SGD(test.parameters(), lr=1)
        opti_test.step()
        
##
#        crit_dekrypt = nn.MSELoss()
#        loss_dekrypt = crit_dekrypt(m_tens, message_out)
#        dekrypt.zero_grad()
#        loss_dekrypt.backward()
#        opti_dekrypt = optim.SGD(dekrypt.parameters(), lr=0.5)
#        opti_dekrypt.step()

#        crit_spy = nn.MSELoss()
#        loss_spy = crit_spy(target, message_out)
#        spy.zero_grad()
#        loss_spy.backward()
#        opti_spy = optim.SGD(spy.parameters(), lr=0.5)
#        opti_spy.step()
        #output= tensor2word(tens)
        
#
        print('Antwort: '+tensor2word(message_out))
if __name__ == "__main__":
    main()
