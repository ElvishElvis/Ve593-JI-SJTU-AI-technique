import numpy as np
import copy
import math
from functools import reduce


class Conv(object):
    def __init__(self, shape, filter_num, filtersize=3, stride=1, padding='VALID'):
        """filter size should be odd"""
        self.shape_input = shape #(batchsize, length, height, channelnum)
        self.filter_num = filter_num
        self.filtersize = filtersize
        self.stride = stride
        self.method = padding
        self.shape_output = list(shape)
        self.shape_output[-1] = filter_num # output channel num = filter num
        if padding != 'SAME':
            self.shape_output[1] =  (self.shape_output[1] - filtersize) // stride +1
            self.shape_output[2] =  (self.shape_output[2] - filtersize) // stride +1
        else:
            self.shape_output[1] =  math.ceil((self.shape_input[1] - filtersize) / stride) +1
            self.shape_output[2] =  math.ceil((self.shape_input[2] - filtersize) / stride) +1
        
        weights_scale = math.sqrt((filtersize**2)*shape[-1]/2)
        self.weights = np.random.standard_normal((filtersize,filtersize,shape[-1],filter_num)) / weights_scale
        self.biases = np.random.standard_normal(filter_num) / weights_scale
        

    def forward(self, x):
        self.x = x
        self.x_col = []
        if self.method == "SAME":
            double_padding = ((self.shape_output[1]-1) * self.stride - (self.shape_input[1] - self.filtersize)) 
            if double_padding % 2 == 0:
                padding1 = (double_padding//2, double_padding//2)
            else:
                padding1 = ((double_padding-1)//2, (double_padding+1)//2)
            double_padding = ((self.shape_output[2]-1) * self.stride - (self.shape_input[2] - self.filtersize)) 
            if double_padding % 2 == 0:
                padding2 = (double_padding//2, double_padding//2)
            else:
                padding2 = ((double_padding-1)//2, (double_padding+1)//2)
      
            x = np.pad(x,((0, 0), padding1, padding2, (0, 0)),'constant')
            self.padding = (padding1, padding2)    
        

        result = np.zeros(tuple(self.shape_output))
        for i in range(self.shape_output[0]):
            self.x_col.append(self.conv(x[i], result[i]))
            #print("res",result[i])
        self.x_col = np.array(self.x_col)
        return result
        
            
    def conv(self, img, maps, filters = None, stride = None, direction = 'FORWARD'):
        if not isinstance(filters,np.ndarray):
            filters = self.weights
        if stride == None:
            stride = self.stride
        if direction == 'FORWARD':
            channel_out = self.filter_num
            channel_in = self.shape_input[-1]
        else:
            channel_in = self.filter_num
            channel_out = self.shape_input[-1]
        
        img_cols = None
        
        for i in range(channel_out):
            if direction == 'FORWARD':
                conv_filter = filters[:,:,:,i]
                conv_bias = self.biases[i]
            else:
                conv_filter = filters[:,:,i,:]
                conv_bias = 0            
            
            conv_map, img_cols = self.convolve(img[:, :, 0], conv_filter[:, :, 0], conv_bias, stride)

            for j in range(1, channel_in):
                new_conv_map, img_col = self.convolve(img[:, :, j], conv_filter[:, :, j], conv_bias, stride)
                conv_map = conv_map + new_conv_map
                img_cols = np.append(img_cols, img_col, axis=1)
                    
            maps[:,:,i] = conv_map
            return img_cols
    
    def convolve(self, img, conv_filter, conv_bias, stride):
        result = np.zeros((img.shape))
        filter_size = conv_filter.shape[1]
        half = int(filter_size / 2)
        img_col = []
        for y in range(half, img.shape[0] - half, stride):
            for x in range(half, img.shape[1] - half, stride): # (x,y) is the position of the center of filter area
                region = img[(y-half):(y+half+1), (x-half):(x+half+1)]
                img_col.append(region.reshape(-1))
                result[y,x] = np.sum(region * conv_filter + conv_bias)
        result = result[half:y+1, half:x+1]
        
        if stride > 1:
            i = 0
            while i < result.shape[0]-1:
                result = np.delete(result,list(range(i+1,i+stride)),axis=0)
                i += 1
            i = 0
            while i < result.shape[1]-1:
                result = np.delete(result,list(range(i+1,i+stride)),axis=1)
                i += 1        
            
        return result,np.array(img_col)
    
    def backward(self, deltas, alpha, lmbda = 0):
        pad_deltas = copy.deepcopy(deltas)
        deltas_prime = np.reshape(deltas, [self.shape_output[0], -1, self.shape_output[-1]])
        
        if(self.stride > 1):
            i = 1
            while(i < pad_deltas.shape[1]):
                for j in range(self.stride-1):
                    v = np.zeros((1,pad_deltas.shape[2],pad_deltas.shape[3]))
                    pad_deltas = np.insert(pad_deltas,i,v,axis=1)
                i += self.stride
            i = 1
            #print(deltas[0,:,:,0])
            while(i < pad_deltas.shape[2]):
                for j in range(self.stride-1):
                    v = np.zeros((1,pad_deltas.shape[3]))
                    pad_deltas = np.insert(pad_deltas,i,v,axis=2)
                i += self.stride
        
        pad_deltas = np.pad(pad_deltas, ((0, 0), (self.filtersize-1, self.filtersize-1), (self.filtersize-1, self.filtersize-1), (0, 0)),'constant')
        weights_flipped = np.zeros(self.weights.shape)
        for i in range(self.weights.shape[-1]):
            for j in range(self.weights.shape[-2]):
                weights_flipped[:,:,j,i] = self.weights[:,:,j,i][::-1]
                for k in range(self.weights.shape[0]):
                    weights_flipped[:,:,j,i][k] = weights_flipped[:,:,j,i][k][::-1]

  
        shape = list(pad_deltas.shape)
        shape[1] = shape[1]-self.filtersize + 1
        shape[2] = shape[2]-self.filtersize + 1
        shape[-1] = self.shape_input[-1]
        result = np.zeros(shape)

        for i in range(self.shape_output[0]):
            self.conv(pad_deltas[i], result[i], weights_flipped, 1, 'BACKWARD')
            #print("res",result[i])
        #print(result[0,:,:,0])
        if self.method != 'SAME':
            z = np.zeros(self.shape_input)
            z[:,0:result.shape[1],0:result.shape[2],:] = result
            result = z
        else:
            (padding1,padding2) = self.padding
            (a,b) = padding1
            b = result.shape[1]-b
            (c,d) = padding2
            d = result.shape[2]-d
            result = result[:,a:b,c:d,:]
        

        ## update ##
        deltas_w = np.zeros(self.weights.shape)
        deltas_b = np.zeros(self.biases.shape)
        n = self.shape_output[0]
        
        for i in range(n):
            deltas_w += np.dot(self.x_col[i].T, deltas_prime[i]).reshape(self.weights.shape)
        deltas_b += np.sum(deltas_prime, axis=(0,1))
        #print(self.weights[:,:,0,0])
        self.weights *= 1 - lmbda
        self.weights -= alpha * deltas_w
        #print(self.weights[:,:,0,0])
        self.biases -= lmbda * deltas_b

        return result
        



class Conv2D(object):
    def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
        self.input_shape = shape
        self.batchsize = shape[0]
        self.stride = stride
        self.ksize = ksize
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.method = method

        

        factorial = 1

        for i in range(len(shape)):
            factorial = factorial * shape[i]

        weights_scale = math.sqrt(factorial / self.output_channels)

        self.weights = np.random.standard_normal(
            (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

        if method == 'SAME':
            pair2 = (shape[0], shape[1] / self.stride, shape[2] / self.stride, self.output_channels)
            self.eta = np.zeros(pair2)

        if method == 'VALID':
            pair1 = (shape[0], (shape[1] - ksize + 1) // self.stride, (shape[1] - ksize + 1) // self.stride,
                                 self.output_channels)
            self.eta = np.zeros(pair1)
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape
        shape_1_mode = shape[1] - ksize // stride
        shape_1_residual = shape[1] - ksize - shape_1_mode * stride
        if shape_1_residual != 0:
            print('''Can't find suitable stride due to invalid tensor width''')
        shape_2_mode = shape[2] - ksize // stride
        shape_2_residual = shape[2] - ksize - shape_2_mode * stride
        if shape_2_residual != 0:
            print('''Can't find suitable stride due to invalid tensor height ''')

    def forward(self, x):
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'SAME':
            x = np.pad(x, ((0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                       'constant', constant_values=0)

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        i = 0
        while (i<self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = self.im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
            i=i+1
        self.col_image = np.array(self.col_image)
        return conv_out

    def backward(self, eta, alpha=0.00001, weight_decay=0.0004):
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array([self.im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)

        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        return next_eta


    def im2col(self,image, ksize, stride):
        # image is a 4d tensor([batchsize, width ,height, channel])
        image_col = []
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
                image_col.append(col)
        image_col = np.array(image_col)

        return image_col



        

class MaxPooling(object):
    def __init__(self, shape, filtersize=2, stride=2, padding = 'VALID'):
        self.shape_input = shape
        self.filtersize = filtersize
        self.stride = stride
        self.shape_output = list(shape)
        if padding != 'SAME':
            self.shape_output[1] =  (self.shape_input[1] - filtersize) // stride +1
            self.shape_output[2] =  (self.shape_input[2] - filtersize) // stride +1
        else:
            self.shape_output[1] =  math.ceil((self.shape_input[1] - filtersize) / stride) +1
            self.shape_output[2] =  math.ceil((self.shape_input[2] - filtersize) / stride) +1
        self.indices = np.zeros((shape[0],self.shape_output[1],self.shape_output[2],shape[-1],2))
        
        
    def forward(self, x):
        result = np.zeros(self.shape_output)
        for i in range(x.shape[0]):
            for j in range(self.shape_input[-1]):
                r_prime = 0
                for r in range(0,self.shape_output[1]*self.stride, self.stride):
                    c_prime = 0
                    for c in range(0,self.shape_output[2]*self.stride, self.stride):
                        result[i,r_prime,c_prime,j] = np.max(x[i,r:(r+self.filtersize), c:(c+self.filtersize),j])
                        index = np.argmax(x[i,r:(r+self.filtersize), c:(c+self.filtersize),j])
                        self.indices[i,r_prime,c_prime,j] = np.array([r + index//self.stride, c + index%self.stride])
                        c_prime += 1
                    r_prime += 1
        return result
    
    def backward(self, deltas):
        result = np.zeros(self.shape_input)
        for i in range(deltas.shape[0]):
            for j in range(self.shape_input[-1]):
                for y_prime in range(deltas.shape[1]):
                    for x_prime in range(deltas.shape[2]):
                        (y,x) = self.indices[i,y_prime,x_prime,j]
                        y = int(y)
                        x = int(x)
                        result[i,y,x,j] = deltas[i,y_prime,x_prime,j]
            
        return result

class activation(object):
    def __init__(self, shape,mode):
        self.deltas = np.zeros(shape)
        self.x = np.zeros(shape)
        self.shape_output = shape
        self.mode=mode

    def forward(self, x):
        if self.mode=="ReLu":
            self.x = x
            return np.maximum(x, 0)
        elif self.mode=="LReLu":
            self.x = x
            return np.maximum(x, 0)+0.5* np.minimum(x,0)
        elif self.mode=="Elu":
            self.x = x
            return np.maximum(x, 0)+0.5*(np.exp(np.minimum(x,0))-1)
        else:
            raise Exception('No such method!')
    def backward(self, deltas):
        if self.mode=="ReLu":
            self.detas = deltas
            self.deltas[self.x<0]=0
            return self.deltas
        elif self.mode=="LReLu":
            self.detas = deltas
            self.deltas[self.x<0]*=0.5
            return self.deltas
        elif self.mode=="Elu":
            self.detas = deltas
            self.deltas[self.x<=0]*=(0.5*np.exp(self.x[self.x<=0]))
            return self.deltas
        else:
            raise Exception('No such method!')






class FC(object):
    def __init__(self, shape, output_num = 2):
        self.shape_input = shape
        self.biases = np.random.standard_normal(output_num)/100
        input_num = reduce(lambda x, y: x * y, shape[1:])
        self.weights = np.random.standard_normal((input_num, output_num))/100
        #print(self.weights)
        self.shape_output = [shape[0], output_num]
    
    def forward(self, x):
        self.x = x.reshape([self.shape_input[0], -1])
        result = np.dot(self.x, self.weights)+self.biases
        return result
 
    def backward(self, deltas, alpha, lmbda = 0.0001):
		## update ##
        deltas_w = np.zeros(self.weights.shape)
        deltas_b = np.zeros(self.biases.shape)
        n = self.shape_output[0]
        
        for i in range(n):
            deltas_w += np.dot(self.x[i][:, np.newaxis], deltas[i][:, np.newaxis].T)
            deltas_b += deltas[i].reshape(self.biases.shape)
        
        self.weights *= (1 - lmbda)
        self.weights -= alpha * deltas_w
        #print(self.weights[:,:,0,0])
        self.biases -= lmbda * deltas_b

        result = np.dot(deltas, self.weights.T)
        result = np.reshape(result, self.shape_input)
        
        return result

class Softmax(object):
    def __init__(self, shape):
        self.shape_input = shape #[batchsize, num]

    def cross_entropy(self, prediction, validation):
        self.validation = validation
        self.prediction = prediction
        self.predict(prediction)
        self.loss = 0
        for i in range(self.shape_input[0]):
            self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, validation[i]]

        return self.loss

    def predict(self, x):
        self.result = np.zeros(x.shape)
        x_exp = np.zeros(x.shape)
        for i in range(self.shape_input[0]):
            x[i, :] -= np.max(x[i, :]) # avoid overflow
            #x_exp[i] = np.exp(x[i])
            self.result[i] = np.exp(x[i])/np.sum(np.exp(x[i]))
        return self.result
    
    def backward(self):
        self.deltas = self.result.copy()
        for i in range(self.shape_input[0]):
            self.deltas[i, self.validation[i]] -= 1
        return self.deltas



