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
from torch.autograd import Variable


###############################################################################
###############################################################################
class Krypt(nn.Module):
    def __init__(self):
        super(Krypt, self).__init__()
        self.lin1 = nn.Linear(50, 1000)
        self.lin2 = nn.Linear(1000, 500)
        self.lin3 = nn.Linear(500, 50)

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


class Spy(nn.Module):
    def __init__(self):
        super(Spy, self).__init__()
        self.lin1 = nn.Linear(50, 500)
        self.lin2 = nn.Linear(500, 1000)
        self.lin3 = nn.Linear(1000, 50)

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
        self.lin1 = nn.Linear(50, 100)
        self.lin2 = nn.Linear(100, 500)
        self.lin3 = nn.Linear(500, 50)

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
def word2tensor(word):
    for i in word:
        
        
    
    return tensor
    
def tensor2word():
    
    return word






def main():

    krypt = Krypt()
    dekrypt = DeKrypt()
    spy = Spy()

    input = Variable(torch.randn(10, 50))
    for i in range(200):
#        print("\n\nMessage Input:\n")
#        print(input)

        output = krypt(input)
#        print(output)

        message_out = dekrypt(output)
#        print("\n\nMessage Output:\n")
#        print(message_out)

        bugging = spy(output)
#        print("\n\nBugging:\n")
#        print(bugging)

        ###Test
        targ_tens = torch.zeros(10, 50)
        targ_tens[0][0] = 1
        targ_tens[1][17] = 1
        targ_tens[2][13] = 1
        targ_tens[3][4] = 1
        target = Variable(targ_tens)
#        print(target)
        ###

#        crit_krypt = nn.MSELoss()
#        loss_krypt = crit_krypt(target, message_out)
#        krypt.zero_grad()
#        loss_krypt.backward()
#        opti_krypt = optim.SGD(krypt.parameters(), lr=0.5)
#        opti_krypt.step()

        crit_dekrypt = nn.MSELoss()
        loss_dekrypt = crit_dekrypt(target, message_out)
        dekrypt.zero_grad()
        loss_dekrypt.backward()
        opti_dekrypt = optim.SGD(dekrypt.parameters(), lr=0.5)
        opti_dekrypt.step()

#        crit_spy = nn.MSELoss()
#        loss_spy = crit_spy(target, message_out)
#        spy.zero_grad()
#        loss_spy.backward()
#        opti_spy = optim.SGD(spy.parameters(), lr=0.5)
#        opti_spy.step()

    print(output)
if __name__ == "__main__":
    main()
