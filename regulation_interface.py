import numpy as np

class RegulationInterface(object):
    regulations = ['L1', 'L2']
    @staticmethod
    def norm_reg(weight, reg, regulation):
        if regulation == 'L2':
            return np.sum(weight*weight)*reg/2
        if regulation == 'L1':
            return np.sum(np.abs(weight))*reg
    @staticmethod    
    def dnorm_reg(weight, reg, regulation):
        if regulation == 'L2':
            return weight*reg
        if regulation == 'L1':
            return np.sign(weight)*reg
    @staticmethod
    def check_regulation(regulation):
        if regulation not in RegulationInterface.regulations:        
            raise ValueError('''Regulation methods: L1, L2!''')