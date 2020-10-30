#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:19:39 2018

@author: apple
"""

import numpy as np

class ActivationInterface(object):
    activations = ['ReLU', 'ELU']
    @staticmethod
    def activation(data, activation):
        if activation == 'ReLU':
            data = np.maximum(0, data)
            return data
        if activation == 'ELU':
            expdata = np.exp(data) - 1
            data = np.where(data > 0, data, expdata) 
            return data
    @staticmethod    
    def dactivation(ddata, data, activation):
        if activation == 'ReLU':
            ddata[data <= 0] = 0
            return ddata
        if activation == 'ELU':
            ddatatemp = ddata*(data+1)
            ddata = np.where(data > 0, ddata, ddatatemp) 
            return ddata
    @staticmethod
    def check_activation(activation):
        if activation not in ActivationInterface.activations:        
            raise ValueError('''Activation methods: ReLU, ELU!''')
            
  
            