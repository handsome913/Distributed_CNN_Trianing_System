import numpy as np
from activation_interface import ActivationInterface
from  LT_code_distribute.server import win_server_run
'''
#-------------test------------
from vgg_net import VGGNet
#from cnn_block_interface import CnnBlockInterface
from cnn_train_interface import CnnTrainInterface
from optimizer_interface import OptimizerInterfaceMDS
from regulation_interface import RegulationInterface
from MNIST_interface import MNISTInterface
#-------------test------------
'''
class CnnBlockInterface(ActivationInterface):
    '''
    the implementation of three basic blocks of cnn net:
    the conv pool and fc block
    and the softmax layer
    '''
    @staticmethod
    def conv_layer(in_data, weights, biases, layer_param=(0,3,1,1), activation='ReLU'):        
        '''
        in_data.shape = [batch,in_height,in_width,in_depth]        
        weights.shape = [filter_size*filter_size*in_depth, out_depth]     
        biases.shape = [1, out_depth]
        
        out_data.shape = [batch,out_height,out_width,out_depth]
        the data for calu gradient: matric_data, filter_data
        '''
        (batch, in_height, in_width, in_depth) = in_data.shape
        (out_depth, filter_size, stride, padding) = layer_param
        if padding:
            padding_data = np.zeros((batch, in_height + 2*padding, in_width + 2*padding, in_depth) )
            padding_data[:, padding : -padding, padding : -padding, :] = in_data
        else:
            padding_data = in_data
            
        filter_size2 = filter_size*filter_size
        height_ef = padding_data.shape[1] - filter_size + 1
        width_ef = padding_data.shape[2] - filter_size + 1
        
        out_height = (in_height - filter_size + 2*padding)//stride + 1
        out_width = (in_width - filter_size + 2*padding)//stride + 1
        out_size = out_height*out_width       
        matric_data = np.zeros( (out_size*batch, filter_size2*in_depth) )

        for i_batch in range(batch):
            i_batch_size = i_batch*out_size
            for i_h, i_height in zip(range(out_height), range(0, height_ef, stride)):
                i_height_size = i_batch_size + i_h*out_width
                for i_w, i_width in zip(range(out_width), range(0, width_ef, stride)):
                    matric_data[i_height_size + i_w, :] = padding_data[i_batch,  i_height : i_height + filter_size,
                                                                       i_width : i_width  + filter_size, :].ravel()        
        print("fp_conv",matric_data.shape,"*",weights.shape)
        #filter_data = np.dot(matric_data, weights) + biases

        filter_data= win_server_run(matric_data,weights)+ biases
        filter_data = CnnBlockInterface.activation(filter_data, activation)  
        
        out_data = np.zeros((batch, out_height, out_width, out_depth))

        for i_batch in range(batch):
            i_batch_size = i_batch*out_size
            for i_height in range(out_height):
                i_height_size = i_batch_size + i_height*out_width
                for i_width in range(out_width):
                    out_data[i_batch, i_height, i_width, :] = filter_data[i_height_size + i_width, :]

        return (matric_data, filter_data, out_data)
    
    @staticmethod
    def dconv_layer(dout_data, matric_data, filter_data, weights, maps_shape, layer_param=(3,1,1), activation='ReLU'):
        '''
        inputs: dout_data, matric_data, filter_data
        matric_data, filter_data are data produced in the forward
        outputs: (dweight, dbias, din_data)
        '''
        (filter_size, stride, padding) = layer_param
        (in_height, in_width, in_depth) = maps_shape
        (batch, out_height, out_width, out_depth) = dout_data.shape
        out_size = out_height*out_width        
        dfilter_data = np.zeros_like(filter_data)
        
        for i_batch in range(batch):
            i_batch_size = i_batch*out_size
            for i_height in range(out_height):
                i_height_size = i_batch_size + i_height*out_width
                for i_width in range(out_width):
                    dfilter_data[i_height_size + i_width, :] = dout_data[i_batch, i_height, i_width, :]
        
        dfilter_data = CnnBlockInterface.dactivation(dfilter_data, filter_data, activation)

        #backprop the dot product filter_data = np.dot(matric_data, weights) + biases
        print("bp_dconv",matric_data.T.shape,"*",dfilter_data.shape)
        #dweight = np.dot(matric_data.T, dfilter_data)
        dweight= win_server_run(matric_data.T, dfilter_data)
        dbias = np.sum(dfilter_data, axis=0, keepdims=True)
        print("bp_dconv",dfilter_data.shape,"*",weights.T.shape)
        #dmatric_data = np.dot(dfilter_data, weights.T)
        dmatric_data = win_server_run(dfilter_data, weights.T)
        #backprop the dmatric_data to dpadding_data, just change the shape.
        padding_height = in_height + 2*padding
        padding_width = in_width + 2*padding
        dpadding_data = np.zeros((batch, padding_height, padding_width, in_depth) )
        
        height_ef = padding_height - filter_size + 1
        width_ef = padding_width - filter_size + 1

        for i_batch in range(batch):
            i_batch_size = i_batch*out_size
            for i_h, i_height in zip(range(out_height), range(0, height_ef, stride)):
                i_height_size = i_batch_size + i_h*out_width
                for i_w, i_width in zip(range(out_width), range(0, width_ef, stride)):
                    dpadding_data[i_batch, i_height : i_height + filter_size, i_width : i_width  + filter_size, :] += dmatric_data[i_height_size + i_w, :].reshape(filter_size, filter_size, -1) 
                    
        #backprop the dpadding_data to din_data
        if padding:
            din_data = dpadding_data[:,padding:-padding,padding:-padding,:]
        else:
            din_data = dpadding_data          
        return (dweight, dbias, din_data)
    
    @staticmethod
    def pooling_layer(in_data, filter_size=2, stride=2):
        '''
        in_data.shape = [batch,in_height,in_width,in_depth]
        
        out_data.shape = [batch,out_height,out_width,out_depth=in_depth]
        the data for calu gradient: matric_data_max_pos
        '''
        (batch, in_height, in_width, in_depth) = in_data.shape
        filter_size2 = filter_size*filter_size
        height_ef = in_height - filter_size + 1
        width_ef = in_width - filter_size + 1
        out_height = (in_height - filter_size)//stride + 1 
        out_width = (in_width - filter_size)//stride + 1 
        out_size = out_height*out_width        
        matric_data = np.zeros( (out_size*in_depth*batch, filter_size2) )

        for i_batch in range(batch):
            i_batch_size = i_batch*out_size*in_depth
            for i_h, i_height in zip(range(out_height), range(0, height_ef, stride)):
                i_height_size = i_batch_size + i_h*out_width*in_depth
                for i_w, i_width in zip(range(0, in_depth*out_width, in_depth), range(0, width_ef, stride)):
                    md = matric_data[i_height_size + i_w : i_height_size + i_w + in_depth, : ]
                    src = in_data[i_batch, i_height : i_height + filter_size, i_width : i_width + filter_size, :]
                    for i in range(filter_size):
                        for j in range(filter_size):
                            md[:, i*filter_size + j] = src[i, j, :]
                            
        matric_data_max_value = matric_data.max(axis = 1, keepdims = True)
        matric_data_max_pos = matric_data == matric_data_max_value #for calu grad
               
        out_depth = in_depth        
        out_data = np.zeros((batch, out_height, out_width, out_depth))

        for i_batch in range(batch):
            i_batch_size = i_batch*out_size*out_depth
            for i_height in range(out_height):
                i_height_size = i_batch_size + i_height*out_width*out_depth
                for i_width in range(out_width):
                    out_data[i_batch, i_height, i_width, :] = matric_data_max_value[i_height_size + i_width*out_depth :
                                            					i_height_size + i_width*out_depth + out_depth].ravel()             
        return (out_data, matric_data_max_pos) 
    
    @staticmethod
    def dpooling_layer(dout_data, matric_data_max_pos, maps_shape, filter_size=2, stride=2):
        '''
        dout_data.shape = [batch,out_height,out_width,out_depth=in_depth]
        matric_data_max_pos.shape = [batch,in_height,in_width,in_depth]
        
        din_data.shape = [batch,in_height,in_width,in_depth]        
        '''               
        (in_height, in_width, not_used) = maps_shape        
        matric_data_not_max_pos =  ~matric_data_max_pos
        (batch, out_height, out_width, in_depth) = dout_data.shape
        out_size = out_height*out_width 
        din_data = np.zeros((batch, in_height, in_width, in_depth), dtype = np.float64)

        height_ef = in_height - filter_size + 1
        width_ef = in_width - filter_size + 1          

        for i_batch in range(batch):
            i_batch_size = i_batch*out_size*in_depth
            for i_h_out, i_height in zip(range(out_height), range(0, height_ef, stride)):
                i_height_size = i_batch_size + i_h_out*out_width*in_depth
                for i_w_dout, i_w, i_width in zip(range(out_width), range(0, in_depth*out_width, in_depth),
                                                  			range(0, width_ef, stride)):
                    md = matric_data_not_max_pos[i_height_size + i_w : i_height_size + i_w + in_depth, : ]
                    din = din_data[i_batch, i_height : i_height + filter_size, i_width : i_width + filter_size, :]
                    dout = dout_data[i_batch, i_h_out, i_w_dout, :]
                    for i in range(filter_size):
                        for j in range(filter_size):
                            din[i, j, :] = dout[:]
                            din[i, j, :][md[:, i*filter_size + j]] = 0                          
        return din_data
    
    @staticmethod
    def FC_layer(in_data, weights, biases, out_depth, last, activation='ReLU'):
        '''
        in_data.shape = [batch, in_height, in_width, in_depth]      
        weights.shape = [filter_size*filter_size*in_depth, out_depth]        
        biases.shape = [1, out_depth]
        last=1 if the FC is the last one        
                
        out_data.shape = [batch,out_height,out_width,out_depth] 
        the data for calu gradient: matric_data, filter_data
        '''        
        (batch, in_height, in_width, in_depth) = in_data.shape              
        matric_data = np.zeros( (batch, in_height*in_width*in_depth) )
        for i_batch in range(batch):
            matric_data[i_batch] = in_data[i_batch].ravel()
        print("fp_FC",matric_data.shape,"*",weights.shape)                
        filter_data = np.dot(matric_data, weights) + biases
        #filter_data = win_server_run(matric_data, weights) + biases        
        if not last: #the last layer not need RELU
            filter_data = CnnBlockInterface.activation(filter_data, activation)

        out_data = np.zeros((batch, 1, 1, out_depth))
        for i_batch in range(batch):
            out_data[i_batch] = filter_data[i_batch]
        
        return (matric_data, filter_data, out_data)

    @staticmethod
    def dFC_layer(dout_data, matric_data, filter_data, weights, maps_shape, last, activation='ReLU'):
        '''
        inputs: dout_data, matric_data, filter_data
        matric_data, filter_data are data produced in the forward
        outputs: (dweight, dbias, din_data)
        '''
        (in_height, in_width, in_depth) = maps_shape       
        (batch, out_height, out_width, out_depth) = dout_data.shape
        
        dfilter_data = np.zeros_like(filter_data)

        for i_batch in range(batch):
            dfilter_data[i_batch] = dout_data[i_batch].ravel()        
        #backprop the ReLU non-linearity
        if not last:
            dfilter_data = CnnBlockInterface.dactivation(dfilter_data, filter_data, activation)

        #backprop the dot product filter_data = np.dot(matric_data, weights) + biases
        print("bp_dFC",matric_data.T.shape,"*",dfilter_data.shape)
        dweight = np.dot(matric_data.T, dfilter_data)
        #dweight = win_server_run(matric_data.T, dfilter_data)
        dbias = np.sum(dfilter_data, axis=0, keepdims=True)
        print("bp_dFC",dfilter_data.shape,"*",weights.T.shape)
        dmatric_data = np.dot(dfilter_data, weights.T)
        #dmatric_data = win_server_run(dfilter_data, weights.T)
        #backprop the dmatric_data to din_data, just change the shape.
        din_data = np.zeros((batch, in_height, in_width, in_depth) )
        for i_batch in range(batch):
            din_data[i_batch] = dmatric_data[i_batch].reshape(in_height, in_width, -1)
            
        return (dweight, dbias, din_data)
        
    @staticmethod
    def softmax_layer(scores):
        """
        scores.shape = [batch,1,1,in_depth]
        probs.shape = [batch,1,1,in_depth]
        """ 
        scores -= np.max(scores, axis=3, keepdims=True)
        exp_scores = np.exp(scores)+10**(-8) # 数值计算更稳定
        exp_scores_sum = np.sum(exp_scores, axis=3, keepdims=True)
        probs = exp_scores/exp_scores_sum
        return probs
    
    @staticmethod       
    def data_loss(probs, labels):
        """
        labels is array of integers specifying correct class
        probs.shape = [batch,1,1,in_depth]
        """
        probs_correct = probs[range(probs.shape[0]), :, :, labels]
        logprobs_correct = -np.log(probs_correct)
        data_loss = np.sum(logprobs_correct)/labels.shape[0]    
        return data_loss

    @staticmethod
    def evaluate_dscores(probs, labels):
        '''
        probs.shape = [batch,1,1,in_depth]
        labels is array of integers specifying correct class
        dscores.shape = [batch,1,1,in_depth]
        '''
        dscores = probs.copy()
        dscores[range(probs.shape[0]), :, :, labels] -= 1
        dscores /= labels.shape[0]
        return dscores   

    @staticmethod
    def param_init(out_depth, in_depth, filter_size2):
        '''
        filter_size2 = filter_size*filter_size
        weights.shape = [filter_size2*in_depth, out_depth]
        '''                     
        std = np.sqrt(2)/np.sqrt(filter_size2*in_depth)
        weights = std * np.random.randn(filter_size2*in_depth, out_depth)
        biases = np.zeros((1, out_depth))
        return (weights, biases)
'''
#---------------------------test-----------------------------------------------
class VGGTest(MNISTInterface, VGGNet, CnnBlockInterface, CnnTrainInterface, OptimizerInterface, RegulationInterface):
    pass

if __name__ == '__main__':
    
    # window下多进程可能有问题，添加这句话缓解
    freeze_support()
    #-----------------test------------
    #window下绑定调用接口不能直接使用lambda，所以只能先定义函数再绑定
    QueueManager.register('get_task_queue', callable=get_task)
    QueueManager.register('get_result_queue', callable=get_result)
    # 绑定端口和设置验证口令
    global manager
    manager = QueueManager(address=('127.0.0.1', 8001), authkey='qiye'.encode())

    # 启动管理，监听信息通道
    manager.start()

    #-----------------test------------

#    struct = [] #linearity model
#    struct = ['FC_64'] # one hidden layer network
    struct = ['conv_8'] + ['pool'] + ['conv_12']*3 + ['pool']  + ['conv_36']*3  + ['pool'] + ['FC_64']
    vgg = VGGTest(struct)
    #60000张图片的训练集，按7:3分成训练集和验证集
    num_samples = 0.7 
    vgg.load_train_data(num_samples)
    train = 1
    scratch = 1
    if train:
        if scratch:
            vgg.train_random_search(lr=[-2.0, -5.0], reg=[-3, -5], num_try=1, epoch_more=20, batch=50, lr_decay=1, mu=0.9, optimizer='adam', regulation='L2', activation='ReLU') # 超参数随机搜索
        else:
            vgg.train_from_checkpoint(epoch_more=2, checkpoint_fname='checkpoint_(loss_-1.23)_(epoch_4)__[(lr reg)_(-3.0 -4.0)]_ adam L2 ELU.npy')

    else:       
        vgg.test_from_checkpoint('checkpoint_(loss_-1.23)_(epoch_4)__[(lr reg)_(-3.0 -4.0)]_ adam L2 ELU.npy')
#------------------------------------------test----------------------------------------------
'''