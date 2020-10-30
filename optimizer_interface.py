import numpy as np

class OptimizerInterface(object):
    optimizers = [ 'Nesterov', 'adam']
    decay_rate=0.999
    eps = 10**(-8)
    @staticmethod
    def nesterov_momentumGD(lr, param, vparam, dparam, mu=0.9):
        pre_vparam = vparam
        vparam = mu*vparam - lr*dparam
        updata_param = vparam + mu*(vparam - pre_vparam)
        update_ratio= np.sum(np.abs(updata_param))/(np.sum(np.abs(param)) + OptimizerInterface.eps)
        param += updata_param
        return update_ratio      
    @staticmethod
    def adam(lr, param, vparam, cache, dparam, t=1, mu=0.9):
        vparam = mu*vparam + (1-mu)*dparam
        vparamt = vparam/(1 - mu**t)
        cache = OptimizerInterface.decay_rate*cache + (1-OptimizerInterface.decay_rate)*(dparam**2)
        cachet = cache/(1 - OptimizerInterface.decay_rate**t)
        updata_param = -(lr/(np.sqrt(cachet) + OptimizerInterface.eps)) * vparamt
        update_ratio = np.sum(np.abs(updata_param))/(np.sum(np.abs(param)) + OptimizerInterface.eps)
        param += updata_param
        return update_ratio
    @staticmethod
    def check_optimizer(optimizer):
        if optimizer not in OptimizerInterface.optimizers:        
            raise ValueError('''updates methods: Nesterov and adam!''')