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
import torch.nn.functional as F
#import torch.optim as optim
from torch.autograd import Variable


###############################################################################
class Krypt(nn.Module):
    def __init__(self):
        super(Krypt, self).__init__()
        self.lin1 = nn.Linear(50,1000)
        self.lin2 = nn.Linear(1000,500)
        self.lin3 = nn.Linear(500,50)

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


class Spy(nn.Module):
    def __init__(self):
        super(Spy, self).__init__()
        self.lin1 = nn.Linear(50,500)
        self.lin2 = nn.Linear(500,1000)
        self.lin3 = nn.Linear(1000,50)

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


#class DeKrypt(nn.Module):
#    def _init__(self):
#        super(DeKrypt, self).__init__()
#        self.lin1 = nn.Linear(50,500)
#        self.lin2 = nn.Linear(500,1000)
#        self.lin3 = nn.Linear(1000,50)
#
#
#    def forward(self, x):
#        x = F.relu_(self.lin1(x))
#        x = F.sigmoid(self.lin2(x))
#        x = F.tanh(self.lin3(x))
#        return x
#
#    def num_flat_features(self, x):
#        size = x.size()[1:]
#        num_features = 1
#        for s in size:
#            num_features *= s
#        return num_features


###############################################################################
def main():

    krypt = Krypt()
#    dekrypt = DeKrypt()
    spy = Spy()

    input = Variable(torch.randn(50))
    print(input)
    output = krypt(input)
    print(output)

 #   message = dekrypt(output)
    bugging = spy(output)

#    print("Message: " + message + " | Bugging: " + bugging)


###############################################################################

if __name__ == "__main__":
    main()
