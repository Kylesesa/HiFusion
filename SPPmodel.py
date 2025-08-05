import math
import  torch
import torch.nn.functional as F

class SPPLayer(torch.nn.Module):
    '''
    空间池化金字塔，在使用鉴别器时，由于融合结果尺寸不一，造成最后无法拉直，故采用此方法，统一最终大小
    '''
    def __init__(self,num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        B,C,H,W = x.size()
        tensor = []
        x_flatten = []
        #print(x.size)
        for i in range(len(self.num_levels)):
            level = self.num_levels[i]
            # print(self.num_levels)
            kernel_size = (math.ceil(H/level),math.ceil(W/level))
            # print(kernel_size)
            #stride = (math.floor(H/level),math.floor(W/level))
            pooling = (math.floor((kernel_size[0]*level-H+1)/2), math.floor((kernel_size[1]*level-W+1)/2))
            # print(pooling)

            zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
            x_new = zero_pad(x)

            H_new = 2*pooling[0] + H
            W_new = 2 * pooling[1] + W

            kernel_size = (math.ceil(H_new / level), math.ceil(W_new / level))
            stride = (math.floor(H_new / level), math.floor(W_new / level))

            if self.pool_type =='max_pool':
                tensor = F.max_pool2d(x_new, kernel_size = kernel_size, stride = stride).view(B,-1)

            else:
                tensor = F.avg_pool2d(x_new, kernel_size = kernel_size, stride = stride).view(B,-1)

            if(i==0):
                x_flatten = tensor.view(B,-1)

            else:
                x_flatten = torch.cat((x_flatten, tensor.view(B,-1)),1)

        return x_flatten
