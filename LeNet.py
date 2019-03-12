
import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce
import fetch_MNIST
from PIL import Image

class LeNet(object):
    def __init__(self, lr=0.1):
        self.lr = lr
        self.conv1 = param_init(6, 1, 5, 5)
        self.pool1 = [2, 2]
        self.conv2 = param_init(16, 6, 5, 5)
        self.pool2 = [2, 2]
        self.conv3 = param_init(120, 16, 5, 5)
        self.fc1 = param_init(400, 120, fc=True)
        self.fc2 = param_init(120, 84, fc=True)
        self.fc3 = param_init(84, 10, fc=True)

    def forward_prop(self, input_data):
        print("Input Data Shape : ",input_data.shape)
        self.l0 = np.expand_dims(input_data, axis=1) / 255
        #print(self.l0.shape)
        self.l1 = self.convolution(self.l0, self.conv1,2)     
        print("After 1st Convolution Data Shape : ",self.l1.shape)
        #print(self.l1[0].shape)

        self.l11=self.relu(self.l1)
        self.l12=self.l11[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show() 

        self.l2 = self.maxpool(self.l1, self.pool1)        
        print("After Maxpool Data Shape ",self.l2.shape)

        self.l12=self.l2[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()

        self.l3 = self.convolution(self.l2, self.conv2,0)   
        print("After 2nd Convolution Data Shape : ",self.l3.shape)  

        self.l11=self.relu(self.l3)
        self.l12=self.l11[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()

        self.l4 = self.maxpool(self.l3, self.pool2)     
        print("After Maxpool Data Shape : ",self.l4.shape)   

        
        self.l12=self.l4[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()

        self.l5= self.convolution(self.l4,self.conv3,0)
        #self.l5 = self.fully_connect(self.l4, self.fc1)  
        print("After 3rd Convolution Data Shape : ",self.l5.shape)  

        self.l11=self.relu(self.l5)
        self.l12=self.l11[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()  

        self.l6 = self.relu(self.l5)                      
        #print(self.l6.shape)   
        self.l7 = self.fully_connect(self.l6, self.fc2)   
        print("After Fully Connected Layer 1 Data Shape : ",self.l7.shape)   
        self.l8 = self.relu(self.l7)
        #print(self.l8.shape)
        self.l9 = self.fully_connect2(self.l8, self.fc3)  
        print("After Fully Connected Layer 2 Data Shape : ",self.l9.shape)                       
        self.l10 = self.softmax(self.l9)     
        #print(self.l10.shape)                 
        return self.l10

    def forward_prop2(self, input_data):
        print("Input Data Shape : ",input_data.shape)
        self.l0 = np.expand_dims(input_data, axis=1) / 255
        #print(self.l0.shape)
        self.l1 = self.convolution(self.l0, self.conv1,2)     
        print("After 1st Convolution Data Shape : ",self.l1.shape)
        #print(self.l1[0].shape)

        self.l11=self.sigmoid(self.l1)
        self.l12=self.l11[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show() 

        self.l2 = self.maxpool(self.l1, self.pool1)        
        print("After Maxpool Data Shape ",self.l2.shape)

        self.l12=self.l2[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()

        self.l3 = self.convolution(self.l2, self.conv2,0)   
        print("After 2nd Convolution Data Shape : ",self.l3.shape)  

        self.l11=self.sigmoid(self.l3)
        self.l12=self.l11[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()

        self.l4 = self.maxpool(self.l3, self.pool2)     
        print("After Maxpool Data Shape : ",self.l4.shape)   

        
        self.l12=self.l4[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()

        self.l5= self.convolution(self.l4,self.conv3,0)
        #self.l5 = self.fully_connect(self.l4, self.fc1)  
        print("After 3rd Convolution Data Shape : ",self.l5.shape)  

        self.l11=self.sigmoid(self.l5)
        self.l12=self.l11[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()  

        self.l6 = self.sigmoid(self.l5)                      
        #print(self.l6.shape)   
        self.l7 = self.fully_connect(self.l6, self.fc2)   
        print("After Fully Connected Layer 1 Data Shape : ",self.l7.shape)   
        self.l8 = self.sigmoid(self.l7)
        #print(self.l8.shape)
        self.l9 = self.fully_connect2(self.l8, self.fc3)  
        print("After Fully Connected Layer 2 Data Shape : ",self.l9.shape)                       
        self.l10 = self.softmax(self.l9)     
        #print(self.l10.shape)                 
        return self.l10

    def forward_prop3(self, input_data):
        print("Input Data Shape : ",input_data.shape)
        self.l0 = np.expand_dims(input_data, axis=1) / 255
        #print(self.l0.shape)
        self.l1 = self.convolution(self.l0, self.conv1,2)     
        print("After 1st Convolution Data Shape : ",self.l1.shape)
        #print(self.l1[0].shape)

        self.l11=self.tanh(self.l1)
        self.l12=self.l11[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show() 

        self.l2 = self.maxpool(self.l1, self.pool1)        
        print("After Maxpool Data Shape ",self.l2.shape)

        self.l12=self.l2[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()

        self.l3 = self.convolution(self.l2, self.conv2,0)   
        print("After 2nd Convolution Data Shape : ",self.l3.shape)  

        self.l11=self.tanh(self.l3)
        self.l12=self.l11[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()

        self.l4 = self.maxpool(self.l3, self.pool2)     
        print("After Maxpool Data Shape : ",self.l4.shape)   

        
        self.l12=self.l4[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()

        self.l5= self.convolution(self.l4,self.conv3,0)
        #self.l5 = self.fully_connect(self.l4, self.fc1)  
        print("After 3rd Convolution Data Shape : ",self.l5.shape)  

        self.l11=self.tanh(self.l5)
        self.l12=self.l11[0].transpose()
        #print(self.l12.shape)
        img1 = Image.fromarray((self.l12[:,:,0]*255).astype(np.uint8))
        img1 = img1.resize((28,28))
        img1.show()  

        self.l6 = self.tanh(self.l5)                      
        #print(self.l6.shape)   
        self.l7 = self.fully_connect(self.l6, self.fc2)   
        print("After Fully Connected Layer 1 Data Shape : ",self.l7.shape)   
        self.l8 = self.tanh(self.l7)
        #print(self.l8.shape)
        self.l9 = self.fully_connect2(self.l8, self.fc3)  
        print("After Fully Connected Layer 2 Data Shape : ",self.l9.shape)                       
        self.l10 = self.softmax(self.l9)     
        #print(self.l10.shape)                 
        return self.l10

    
    def convolution(self, input_map, kernal,pad):
        N, C, W, H = input_map.shape
        K_NUM, K_C, K_W, K_H = kernal.shape
        feature_map = np.zeros((N, K_NUM, W-K_W+2*pad+1, H-K_H+2*pad+1))
        #print(1)
        #print(input_map.shape)
        input_map = np.pad(input_map,((0,0), (0,0), (pad, pad), (pad, pad)),'constant')
        #print(2)
        #print(input_map.shape)
        for imgId in range(N):
            for kId in range(K_NUM):
                for cId in range(C):
                    #print(input_map.shape,kernal.shape,feature_map.shape)
                    feature_map[imgId][kId] += convolve2d(input_map[imgId][cId], kernal[kId,cId,:,:], mode='valid')
        return feature_map
        

    def maxpool(self, input_map, pool):
        N, C, W, H = input_map.shape
        P_W, P_H = tuple(pool)
        feature_map = np.zeros((N, C, W//P_W, H//P_H))
        feature_map = block_reduce(input_map, tuple((1, 1, P_W, P_H)), func=np.max)
        feature_map = input_map.reshape(N, C, W//P_W, P_W, H//P_H, P_H).max(axis=(3,5))
        return feature_map

    def fully_connect(self, input_data, fc):
        N = input_data.shape[0]
        output_data = np.dot(input_data.reshape(N, -1), fc)
        return output_data

    def fully_connect2(self, input_data, fc):
        N = input_data.shape[0]
        #input_data=input_data
        #print(input_data[0])
        y=[]
        for i in range(fc.shape[1]):
            sum=0
            for j in range(fc.shape[0]):
                sum+=(input_data[0][j]-fc[j][i])**2
            y.append(sum)
        #print(y)
        return np.array(y).reshape(1,fc.shape[1])

    def relu(self, x):
        return x * (x > 0)

    def sigmoid(self,x):
	    return 1 / (1+np.exp(-x))

    def tanh(self,x):
	    return np.tanh(x)
        

    def softmax(self, x):
        y = list()
        for t in x:
            e_t = np.exp(t - np.max(t))
            y.append(e_t / e_t.sum())
        return np.array(y)


def param_init(c1, c2, w=1, h=1, fc=False):
    fan_1 = c2 * w * h
    fan_2 = c1 * w * h
    ratio = np.sqrt(6.0 / (fan_1 + fan_2))
    params = ratio * (2*np.random.random((c1, c2, w, h)) - 1)
    if fc == True:
        params = params.reshape(c1, c2)
    return params



def run(val):
    train_imgs = fetch_MNIST.load_test_images()
    train_labs = fetch_MNIST.load_test_labels().astype(int)
    data_size = train_imgs.shape[0]
    lr = 0.01     
    max_iter = 1
    my_CNN = LeNet(lr)
    input_data=train_imgs[0:1]
    img1 = Image.fromarray(input_data[0])
    #mg1 = img1.resize((28,28))
    img1.show()
    import matplotlib.pyplot as plt
    if val==1:
        softmax_output = my_CNN.forward_prop2(input_data)
    elif val==2:
        softmax_output = my_CNN.forward_prop(input_data)
    else:
        softmax_output = my_CNN.forward_prop3(input_data)
    print("Input Image")
    plt.imshow(train_imgs[0],cmap='gray')
    plt.show()
    print("Softmax Output")
    print(int(np.argmax(softmax_output)))
