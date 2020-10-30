#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:18:14 2018

@author: apple
"""


from vgg_net import VGGNet
from cnn_block_interface import CnnBlockInterface
from cnn_train_interface import CnnTrainInterface
from optimizer_interface import OptimizerInterface
from regulation_interface import RegulationInterface
from MNIST_interface import MNISTInterface
class VGGTest(MNISTInterface, VGGNet, CnnBlockInterface, CnnTrainInterface, OptimizerInterface, RegulationInterface):
    pass

if __name__ == '__main__':
    
    '''
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
    #这里添加了client函数.把它放到单独的一个进程中去
    p = Process(target=win_client_run)
    p.start()
    p2 = Process(target=win_client_run)
    p2.start()
    #-----------------test------------
    '''
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
 
#%%
'''
98.73 checkpoint_(loss_-1.23)_(epoch_4)__[(lr reg)_(-3.0 -4.0)]_ adam L2 ELU

只使用随机40个样本，测试集准确率达63.68
400个达到90.05
0.5样本准确率97.15
0.25样本准确率97.49
0.1样本准确率96.34
0.05样本准确率92.93
0.01样本准确率83.25
'''