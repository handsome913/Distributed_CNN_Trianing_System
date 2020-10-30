import numpy as np
import re

class VGGNet(object):
    '''
    only support the VGG like cnn struct
    struct = ['conv_16_5_2_2'] + ['conv_32']*2 + ['pool'] + ['conv_64']*3 + ['pool'] + ['FC_128']
    conv_16_5_2_2 = conv_featureMapNum_[_filterSize=3_stride=1_padding=1]
    the last TWO layers always are: FC, softmax
    all pool layer always have filterSize=2 stride=2    
    surport save/load the checkpoint
    '''           
    def __init__(self, struct=[]):
        if len(struct) == 0:
            print('you are using linearity model!')
        self.__struct_parse(struct)
        self.__struct = struct
        self.__struct += ['FC', 'softmax']
    
    def __struct_parse(self, struct):
        layers = []
        for layer in struct:
            convfull = re.match('^conv_(\d{1,3})_(\d{1})_(\d{1})_(\d{1})$', layer)
            convdefault = re.match('^conv_(\d{1,3})$', layer)
            pool = re.match('^pool$', layer)
            fc = re.match('^FC_(\d{1,4})$', layer)
            if convfull:
                layers.append(( int(convfull.group(1)), int(convfull.group(2)), 
                               int(convfull.group(3)), int(convfull.group(4)), 'conv'))
            elif convdefault:
                layers.append(( int(convdefault.group(1)),3,1,1,'conv'))
            elif pool:
                layers.append( (layers[-1][0], 'pool') )
            elif fc:
                layers.append( (int(fc.group(1)), 'FC') )
            else:
                raise ValueError('the layer must like conv_16_5_2_2 or conv_16 or pool or FC_64')
                       
        layers.append(('', 'Last_FC'))
        self.__layers_params = layers
        
    def featuremap_shape(self):        
        maps_shape = []
        in_map_shape = (self.im_height, self.im_width, self.im_dims)
        maps_shape.append(in_map_shape)
        for layer in self.__layers_params:
            if layer[-1] == 'Last_FC':
                break
            elif layer[-1] == 'FC':
                in_map_shape = (1, 1, layer[0])                              
            elif layer[-1] == 'conv':
                (out_depth, filter_size, stride, padding, not_used) = layer
                out_height = (in_map_shape[0] - filter_size + 2*padding)//stride + 1
                out_width = (in_map_shape[1] - filter_size + 2*padding)//stride + 1
                in_map_shape = (out_height, out_width, out_depth)
                if out_height < filter_size or out_width < filter_size:
                    raise ValueError('the cnn struct is not compatible with the image size!\n')
            elif layer[-1] == 'pool':
                filter_size = 2
                stride = 2
                out_height = (in_map_shape[0] - filter_size)//stride + 1
                out_width = (in_map_shape[1] - filter_size)//stride + 1
                in_map_shape = (out_height, out_width, layer[0])
                if out_height < filter_size or out_width < filter_size:
                    raise ValueError('the cnn struct is not compatible with the image size!\n')   
            else:
                pass
            maps_shape.append(in_map_shape)       
        self.maps_shape = maps_shape
        
    def init_params(self):
        self.__weights = []
        self.__biases = []
        in_depth = self.im_dims
        out_depth = in_depth
        for layer_param, map_shape in zip(self.__layers_params, self.maps_shape):
            weight = np.array([])
            bias = np.array([])
            if layer_param[-1] == 'Last_FC':
                in_depth = out_depth
                out_depth = self.num_class
                (weight, bias) = self.param_init(out_depth, in_depth, map_shape[0]*map_shape[1])
            elif layer_param[-1] == 'FC':
                out_depth = layer_param[0]
                in_depth = map_shape[2]
                (weight, bias) = self.param_init(out_depth, in_depth, map_shape[0]*map_shape[1])
            elif layer_param[-1] == 'conv':
                filter_size = layer_param[1]
                out_depth = layer_param[0]
                (weight, bias) = self.param_init(out_depth, in_depth, filter_size*filter_size)
            elif layer_param[-1] == 'pool': # pool has no params
                pass
            else: 
                pass
            in_depth = out_depth
            self.__weights.append(weight)
            self.__biases.append(bias)
        #softmax layer: no params        
        #for backprop
        self.__vweights = []
        self.__vbiases = []
        self.__cache_biases = []
        self.__cache_weights = []
        for weight, bias in zip(self.__weights, self.__biases):
            self.__vweights.append(np.zeros_like(weight))
            self.__vbiases.append(np.zeros_like(bias))
            self.__cache_weights.append(np.zeros_like(weight))
            self.__cache_biases.append(np.zeros_like(bias))
            
    def reg_loss(self, reg=10**(-5), regulation='L2'):
        reg_loss = 0
        for weight in self.__weights:
            if weight.size != 0:
                reg_loss += self.norm_reg(weight, reg, regulation)
        return reg_loss
    
    def forward(self,batch_data, labels, reg=10**(-5), regulation='L2', activation='ReLU'):
        self.__matric_data = []
        self.__filter_data = []
        self.__matric_data_max_pos = []

        in_maps = batch_data                
        for layer_param, weight, bias in zip(self.__layers_params, self.__weights, self.__biases):
            matric_data = np.array([])
            filter_data = np.array([])
            matric_data_max_pos = np.array([])
            if layer_param[-1] == 'Last_FC': #last FC layer, no non linearity
                (matric_data, filter_data, out_maps) = self.FC_layer(in_maps, weight, bias, self.num_class, 1, activation)
            elif layer_param[-1] == 'FC':               
                (matric_data, filter_data, out_maps) = self.FC_layer(in_maps, weight, bias, layer_param[0], 0, activation)
            elif layer_param[-1] == 'conv':
                (matric_data, filter_data, out_maps) = self.conv_layer(in_maps, weight, bias, layer_param[0:-1], activation)               
            elif layer_param[-1] == 'pool':
                (out_maps, matric_data_max_pos)  = self.pooling_layer(in_maps)
            else:
                pass
            in_maps = out_maps

            self.__matric_data.append(matric_data)
            self.__filter_data.append(filter_data)
            self.__matric_data_max_pos.append(matric_data_max_pos)

        self.__probs = self.softmax_layer(out_maps)
        data_loss = self.data_loss(self.__probs, labels)
        reg_loss = self.reg_loss(reg, regulation)
        return (data_loss, reg_loss)
    
    def predict(self, batch_data, labels, activation='ReLU'):        
        in_maps = batch_data                
        for layer_param, weight, bias in zip(self.__layers_params, self.__weights, self.__biases):           
            if layer_param[-1] == 'Last_FC': #last FC layer, no non linearity
                (matric_data, filter_data, out_maps) = self.FC_layer(in_maps, weight, bias, self.num_class, 1, activation)
            elif layer_param[-1] == 'FC':               
                (matric_data, filter_data, out_maps) = self.FC_layer(in_maps, weight, bias, layer_param[0], 0, activation)
            elif layer_param[-1] == 'conv':
                (matric_data, filter_data, out_maps) = self.conv_layer(in_maps, weight, bias, layer_param[0:-1], activation)             
            elif layer_param[-1] == 'pool':
                (out_maps, matric_data_max_pos)  = self.pooling_layer(in_maps)
            else:
                pass
            in_maps = out_maps
        predicted_class = np.argmax(out_maps, axis=3)
        accuracy = predicted_class.ravel() == labels        
        return np.mean(accuracy)

    
    def dweight_reg(self, reg=10**(-5), regulation='L2'):
        for i in range(len(self.__weights)):
            weight = self.__weights[i]
            if weight.size != 0:
                self.__dweights[-1-i] += self.dnorm_reg(weight, reg, regulation)
            
    def backpropagation(self, labels, reg=10**(-5), regulation='L2', activation='ReLU'):
        dscores = self.evaluate_dscores(self.__probs, labels)
        dout_maps = dscores
        self.__dweights = []
        self.__dbiases = []
        for (layer_param, maps_shape, weight,
             matric_data, filter_data, matric_data_max_pos) in zip(reversed(self.__layers_params),
            reversed(self.maps_shape), reversed(self.__weights),
            reversed(self.__matric_data), reversed(self.__filter_data), reversed(self.__matric_data_max_pos) ):
            if layer_param[-1] == 'Last_FC':
                (dweight, dbias, din_maps) = self.dFC_layer(dout_maps, matric_data, filter_data,
                                                               weight, maps_shape, 1, activation)
            elif layer_param[-1] == 'FC':
                (dweight, dbias, din_maps) = self.dFC_layer(dout_maps, matric_data, filter_data,
                                                               weight, maps_shape, 0, activation)
            elif layer_param[-1] == 'conv':   
                (dweight, dbias, din_maps) = self.dconv_layer(dout_maps, matric_data, filter_data,
                                                               weight, maps_shape, layer_param[1:-1], activation)            
            elif layer_param[-1] == 'pool':  
                dweight = np.array([])
                dbias = np.array([])
                din_maps = self.dpooling_layer(dout_maps, matric_data_max_pos, maps_shape)
            else:
                pass
            dout_maps = din_maps
            self.__dweights.append(dweight)
            self.__dbiases.append(dbias)
        self.__dbatch_data = din_maps #grad of input image batch
        self.dweight_reg(reg, regulation)  

    def params_update(self, lr=10**(-4), t=1, mu=0.9, optimizer='Nesterov'):
        self.update_ratio = []
        if optimizer == 'adam':
            for i in range(len(self.__weights)):
                weight = self.__weights[i]
                bias = self.__biases[i]
                dweight = self.__dweights[-1-i]
                dbias = self.__dbiases[-1-i]
                v_weight = self.__vweights[i]
                v_bias = self.__vbiases[i]
                cache_weight = self.__cache_weights[i]
                cache_bias = self.__cache_biases[i]
                if weight.size != 0:
                    update_ratio_w = self.adam(lr, weight, v_weight, cache_weight, dweight, t, mu)
                    update_ratio_b = self.adam(lr, bias, v_bias, cache_bias, dbias, t, mu)
                    self.update_ratio.append((update_ratio_w,update_ratio_b))
                    
        if optimizer == 'Nesterov':
            for i in range(len(self.__weights)):
                weight = self.__weights[i]
                bias = self.__biases[i]
                dweight = self.__dweights[-1-i]
                dbias = self.__dbiases[-1-i]
                v_weight = self.__vweights[i]
                v_bias = self.__vbiases[i]
                if weight.size != 0:
                    update_ratio_w = self.nesterov_momentumGD(lr, weight, v_weight, dweight, mu)
                    update_ratio_b = self.nesterov_momentumGD(lr, bias, v_bias, dbias, mu)
                    self.update_ratio.append((update_ratio_w,update_ratio_b))

    def save_checkpoint(self, fname):   
        with open(fname, 'wb') as f:
            np.save(f, np.array([3,1,4,1,5,9,2,8,8])) # magic number
            np.save(f, np.array( self.__struct) )
            np.save(f, np.array([self.num_class, self.im_dims, self.im_height, self.im_width]) )
            np.save(f, np.array(self.__layers_params))
            np.save(f, np.array(self.maps_shape))
            np.save(f, np.array(self.context))
            for array in self.__weights:
                np.save(f, array)
            for array in self.__biases:
                np.save(f, array)
            for array in self.__vweights:
                np.save(f, array)
            for array in self.__vbiases:
                np.save(f, array)
            for array in self.__cache_weights:
                np.save(f, array)
            for array in self.__cache_biases:
                np.save(f, array)
 
    def load_checkpoint(self, fname):
        with open(fname, 'rb') as f:
            magic_number = np.load(f)
            if not all(magic_number == np.array([3,1,4,1,5,9,2,8,8])):  # magic number
                raise ValueError('the file format is wrong!\n')
            self.__struct = np.load(f)
            print('\n\nthe net struct is: \n', self.__struct)
            self.num_class, self.im_dims, self.im_height, self.im_width = np.load(f)                
            self.__layers_params = np.load(f)
            self.maps_shape = np.load(f)
            self.context = np.load(f)
            self.__weights=[]
            self.__biases=[]
            for i in range(len(self.__layers_params)):
                array = np.load(f)
                self.__weights.append(array)
            for i in range(len(self.__layers_params)):
                array = np.load(f)
                self.__biases.append(array)
            self.__vweights=[]
            self.__vbiases=[]
            for i in range(len(self.__layers_params)):
                array = np.load(f)
                self.__vweights.append(array)
            for i in range(len(self.__layers_params)):
                array = np.load(f)
                self.__vbiases.append(array)                
            self.__cache_weights=[]
            self.__cache_biases=[]
            for i in range(len(self.__layers_params)):
                array = np.load(f)
                self.__cache_weights.append(array)
            for i in range(len(self.__layers_params)):
                array = np.load(f)
                self.__cache_biases.append(array)                
            print('the struct hyper parameters:\n', self.__layers_params)