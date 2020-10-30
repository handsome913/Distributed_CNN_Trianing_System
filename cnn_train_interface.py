import numpy as np
import time
#import matplotlib.pyplot as plt

class CnnTrainInterface(object):
    '''      
    decay the learning rate every epoch using an exponential rate of lr_decay  
    support learning rate and regularization random search 
    also support train and test from checkpoint
    '''
    def __shuffle_data(self):
        shuffle_no = list(range(self.num_train_samples))
        np.random.shuffle(shuffle_no)
        self.train_labels = self.train_labels[shuffle_no]
        self.train_data = self.train_data[shuffle_no]

        shuffle_no = list(range(self.num_val_samples))
        np.random.shuffle(shuffle_no)
        self.val_labels = self.val_labels[shuffle_no]
        self.val_data = self.val_data[shuffle_no]  
    
    def __train(self, epoch_more=20, lr=10**(-4), reg=10**(-5), batch=50, lr_decay=0.8, mu=0.9, 
                optimizer='Nesterov', regulation='L2', activation='ReLU'):               
        '''
        # 可视化数据损失、训练集和验证集准确率
        plt.close()
        fig=plt.figure('')
        ax=fig.add_subplot(3,1,1)
        ax.grid(True)
        ax2=fig.add_subplot(3,1,2)
        ax2.grid(True)
        ax3=fig.add_subplot(3,1,3)
        ax3.grid(True)
        plt.xlabel( 'log10(lr)=' + str(round((np.log10(lr)),2)) + ' ' +  'log10(reg)=' + str(round((np.log10(reg)),2)), fontsize=14)
        plt.ylabel('                                        update_ratio     accuracy       log10(data loss)', fontsize=14)  
        '''
        epoch = 0   
        val_no = 0
        per_epoch_time = self.num_train_samples//batch
        while epoch < epoch_more:
            losses = 0
            self.__shuffle_data()       
            for i in range(0, self.num_train_samples, batch):
                a =time.time()
                batch_data = self.train_data[i:i+batch,:]                
                labels = self.train_labels[i:i+batch]
                (data_loss, reg_loss) = self.forward(batch_data, labels, reg, regulation, activation)   
                losses += data_loss + reg_loss
                self.backpropagation(labels, reg, regulation, activation)
                self.params_update(lr, per_epoch_time*epoch + i+1, mu, optimizer)
                update_ratio = self.update_ratio[0][0]
                '''
                if i % (batch*20) == 0:
                    ax.scatter(i/self.num_train_samples+epoch, np.log10(data_loss), c='b',marker='.')                    
                    train_accuracy = self.predict(batch_data, labels, activation)                    
                    batch_data_val = self.val_data[val_no:val_no+batch,:]                
                    labels_val = self.val_labels[val_no:val_no+batch]
                    val_accuracy = self.predict(batch_data_val, labels_val, activation)                    
                    val_no += batch
                    if val_no >= self.num_val_samples - batch:
                        val_no = 0
                    ax2.scatter(i/self.num_train_samples+epoch, (train_accuracy), c='r',marker='*')
                    ax2.scatter(i/self.num_train_samples+epoch, (val_accuracy), c='b',marker='.')
                    
                    ax3.scatter(i/self.num_train_samples+epoch, np.log10(update_ratio), c='r',marker='.')
                    plt.pause(0.000001)
                '''
                if i % (batch*20) == 0:                   
                    train_accuracy = self.predict(batch_data, labels, activation)                    
                    batch_data_val = self.val_data[val_no:val_no+batch,:]                
                    labels_val = self.val_labels[val_no:val_no+batch]
                    val_accuracy = self.predict(batch_data_val, labels_val, activation)                    
                    val_no += batch
                    if val_no >= self.num_val_samples - batch:
                        val_no = 0

                    print("********num_train_samples:",i,"#######",self.num_train_samples)
                    print("-----------------train_accuracy:",(train_accuracy),"epoch:",epoch,"-----------------")
                    print("-----------------val_accuracy:",(val_accuracy),"epoch:",epoch,"-----------------")
                    print("-----------------update_ratio:",np.log10(update_ratio),"epoch:",epoch,"-----------------")
                b=time.time()
                print('训练一个batch的时间是',b-a)
                exit()
            epoch += 1              
            '''
            plt.savefig('checkpoint_' + '(loss_' + str(round(np.log10(losses/per_epoch_time),2)) +
                                 ')_(epoch_' + str(round(epoch,2)) + ')_' + '_[(lr reg)_' + '(' + str(round((np.log10(lr)),2)) +
                                 ' ' + str(round((np.log10(reg)),2)) + ')]' + '_' + 
                 ' ' + optimizer + ' '+ regulation + ' ' + activation + '.png')
            '''
            self.context[0] = lr
            self.save_checkpoint('checkpoint_' + '(loss_' + str(round(np.log10(losses/per_epoch_time),2)) +
                                 ')_(epoch_' + str(round(epoch,2)) + ')_' + '_[(lr reg)_' + '(' + str(round((np.log10(lr)),2)) +
                                 ' ' + str(round((np.log10(reg)),2)) + ')]' + '_' + 
                 ' ' + optimizer + ' '+ regulation + ' ' + activation + '.npy')
    
            lr *= lr_decay #decayed every epoch using an exponential rate

        self.test(batch, activation)            
    
    def __methods_check(self, optimizer, regulation, activation):        
        self.check_optimizer(optimizer)
        self.check_regulation(regulation)
        self.check_activation(activation)       
               
    @staticmethod
    def __gen_lr_reg(lr=[0, -6], reg=[-3, -6], num_try=10):
        minlr = min(lr)
        maxlr = max(lr)        
        randn = np.random.rand(num_try*2)
        lr_array = 10**(minlr + (maxlr-minlr)*randn[0: num_try])
             
        minreg = min(reg)
        maxreg = max(reg)
        reg_array = 10**(minreg + (maxreg-minreg)*randn[num_try: 2*num_try])       
        lr_regs =  zip(lr_array, reg_array)
        return lr_regs
    
    def train_random_search(self,lr=[-1, -5], reg=[-1, -5], num_try=10, epoch_more=1,batch=64, lr_decay=0.8, mu=0.9, 
                            optimizer='Nesterov', regulation='L2', activation='ReLU'):        
        self.__methods_check(optimizer, regulation, activation)          
        self.featuremap_shape()    
        lr_regs = self.__gen_lr_reg(lr, reg, num_try)       
        for lr_reg in lr_regs:
            try:
                self.init_params()
                self.context = [*lr_reg, batch, lr_decay, mu, optimizer, regulation, activation]
                self.__train(epoch_more, *lr_reg, batch, lr_decay, mu, optimizer, regulation, activation)
            except KeyboardInterrupt:
                pass   
        
    def train_from_checkpoint(self, epoch_more=10, checkpoint_fname=''):                              
        self.load_checkpoint(checkpoint_fname) 
        [lr, reg, batch, lr_decay, mu, optimizer, regulation, activation] = self.context
        lr = np.double(lr)
        reg = np.double(reg)
        batch = np.int(batch)
        lr_decay = np.double(lr_decay)
        mu = np.double(mu)
        self.__train(epoch_more, lr, reg, batch, lr_decay, mu, optimizer, regulation, activation)
        
    def test_from_checkpoint(self, checkpoint_fname):    
        self.load_test_data()
        self.load_checkpoint(checkpoint_fname) 
        [lr, reg, batch, lr_decay, mu, optimizer, regulation, activation] = self.context        
        batch = np.int(batch)      
        accuracys = np.zeros(shape=(self.test_labels.shape[0],))
        for i in range(0, self.test_labels.shape[0], batch):
            batch_data = self.test_data[i:i+batch,:]
            label = self.test_labels[i:i+batch]
            accuracys[i:i+batch] = self.predict(batch_data, label, activation)
            
        accuracy = np.mean(accuracys)            
        print('the test accuracy: %.5f' % accuracy)
        return accuracy
        
    def test(self, batch, activation):   
        self.load_test_data()
        accuracys = np.zeros(shape=(self.test_labels.shape[0],))
        for i in range(0, self.test_labels.shape[0], batch):
            batch_data = self.test_data[i:i+batch,:]
            label = self.test_labels[i:i+batch]
            accuracys[i:i+batch] = self.predict(batch_data, label, activation)
            
        accuracy = np.mean(accuracys)            
        print('the test accuracy: %.5f' % accuracy)
        return accuracy