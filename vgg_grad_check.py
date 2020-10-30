import numpy as np

from vgg_net import VGGNet
from cnn_block_interface import CnnBlockInterface
from regulation_interface import RegulationInterface

class VGGTest(VGGNet, CnnBlockInterface, RegulationInterface):
    def set_data_pro(self, num_class=4, im_height=32, im_width=32, im_dims=3):        
        self.num_class = num_class
        self.im_height = im_height
        self.im_width = im_width
        self.im_dims = im_dims 
        
    def gen_random_data(self):
            self.num_samples = self.num_class*20
            self.data = np.random.randn(self.num_samples, self.im_height, self.im_width, self.im_dims)        
            self.labels = np.random.randint(self.num_class, size=self.num_samples)
        
    def check_gradient(self, check_weight_or_bias=1, step=10**(-5), reg=10**(-1), regulation='L1', activation='ELU'):          
    #    check_weight_or_bias: 1 for weight, 0 for bias   
        self.set_data_pro()
        self.gen_random_data()     
        self.featuremap_shape()
        self.init_params()            
        for layer in range(len(self.maps_shape)):
            if check_weight_or_bias:
                weight = self._VGGNet__weights[layer]
                if weight.size == 0:
                    continue
                else:
                    row = np.random.randint(weight.shape[0])
                    col = np.random.randint(weight.shape[1])
                    param = weight[row][col]                    
            else:
                bias = self._VGGNet__biases[layer]
                if bias.size == 0:
                    continue
                else:
                    row = np.random.randint(bias.shape[1])
                    param = bias[0][row]                
            
            (data_loss, reg_loss) = self.forward(self.data, self.labels, reg, regulation, activation)
            self.backpropagation(self.labels, reg, regulation, activation)              
            if check_weight_or_bias:
                danalytic = self._VGGNet__dweights[-1-layer][row][col]
            else:
                danalytic = self._VGGNet__dbiases[-1-layer][0][row]
        
            if check_weight_or_bias:
                self._VGGNet__weights[layer][row][col] = param - step
            else:
                self._VGGNet__biases[layer][0][row] = param - step    
            (data_loss1, reg_loss) = self.forward(self.data, self.labels, reg, regulation, activation)
            loss1 = data_loss1 + reg_loss
            
            if check_weight_or_bias:
                self._VGGNet__weights[layer][row][col] = param + step
            else:
                self._VGGNet__biases[layer][0][row] = param + step
            (data_loss2, reg_loss) = self.forward(self.data, self.labels, reg, regulation, activation)
            loss2 = data_loss2 + reg_loss
            dnumeric = (loss2 - loss1)/(2*step)
            
            print(layer, data_loss1, data_loss2)    
            error_relative = np.abs(danalytic - dnumeric)/np.maximum(danalytic, dnumeric)   
            print(danalytic, dnumeric, error_relative)  
            
if __name__ == '__main__':
    #网络结构    
    struct = ['conv_32_5_1_0'] + ['pool'] + ['conv_64'] + ['pool'] + ['conv_128']*2 + ['pool'] + ['conv_256'] + ['FC_100']
    vgg = VGGTest(struct) #创建网络实例
    vgg.check_gradient(check_weight_or_bias=1, step=10**(-5), reg=10**(-50), regulation='L1', activation='ReLU')