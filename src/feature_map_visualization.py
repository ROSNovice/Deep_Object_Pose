# coding: utf-8
from __future__ import print_function
#########################################################################################
    # PIL:
    #   https://pillow-zh-cn.readthedocs.io/zh_CN/latest/handbook/tutorial.html
    #   https://www.cnblogs.com/ocean1100/p/9494640.html
    #   读入为‘RGB’,大小为（W x H）
    #   注意ndarray中是 行row x 列col x 维度dim 所以行数是高，列数是宽

    # OPENCV
    #   读入为 “BGR”，类型为 ndarray, (H x W x C)
    #   图片显示问题： https://www.jianshu.com/p/9bc09c4441c5

    # numpy
    #   http://blog.sciencenet.cn/blog-3031432-1064033.html
    #   slicing & indexing: http://www.zmonster.me/2016/03/09/numpy-slicing-and-indexing.html
    #   ndarray.flags: 内存信息：http://www.runoob.com/numpy/numpy-array-attributes.html
    #   numpy.unique： https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.unique.html
    #   numpy.ndarray.view: https://blog.csdn.net/wangwenzhi276/article/details/53428353
    #                       https://blog.csdn.net/Jfuck/article/details/9464959
    #   C_continuous:  https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays

    # PIL图像在转换为numpy.ndarray后，格式为(h,w,c)，像素顺序为RGB；
    # OpenCV在cv2.imread()后数据类型为numpy.ndarray，格式为(h,w,c)，像素顺序为BGR。
#########################################################################################

# from PIL import Image
# import numpy as np 
# image = Image.open("000000.left.jpg")
# print(image.size)
# image.show()
# image = np.array(image,dtype = np.float32)
# print(image.shape)

#########################################################################################

# import cv2
# image = cv2.imread("000000.left.jpg")
# #print(image.size)
# image = cv2.resize(image,(640,480))   # resize(image,(w,h))
# # show image
# cv2.imshow("image",image)
# cv2.waitKey(2)
# # BGR ——> RGB
# image = image[...,: : -1]
# # show image
# cv2.imshow("image",image)
# cv2.waitKey(2)
# # H W C ——> C H W
# image = image.transpose(2,0,1)

########################################################################################
    # https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html
    # Pytorch torchvision.utils.make_grid()用法:只能批量显示图像（化为3通道）：
    #           https://pytorch.org/docs/stable/torchvision/utils.html   
    #           https://blog.csdn.net/u012343179/article/details/83007296
    # cpu、gpu、variable、numpy转换： 
    #           https://blog.csdn.net/qq_16234613/article/details/80025832
    #           https://blog.csdn.net/qq_38410428/article/details/8297371182973711
########################################################################################

import cv2
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt

def save_feature_map_to_image(beliefs, affinities, object_name):
    '''
    print ("the shape of belief map is {}; 
            the shape of affinities map is {}"
            .format(beliefs.shape,a.ffinities.shape))
    '''
    # concat the beliefs map and affinities map
    cat_map_ten = torch.cat((beliefs,affinities), 1) 
    # delete the dimension of batchsize
    cat_map_ten = cat_map_ten.squeeze(0)
    # reshape the channal and to numpy
    temp = []
    for i in range(cat_map_ten.shape[0]):
        # the data form variable type to cpu type and to numpy type
        T = cat_map_ten[i, ...].data.cpu().numpy()
        # C H W ——> H W C
        T = np.array([T,T,T]).transpose(1,2,0)
        temp.append(T)
    cat_map_arr = np.array(temp)
    # process cat_map_arr
    # cat_map_arr = map_process(cat_map_arr)
    # save_image
    output_image = plt.figure()
    for i in range(cat_map_ten.shape[0]):
        plt.subplot(3,9,i+1)
        if i < 9:
            plt.title('b{}'.format(i))
        else:
            plt.title('a{}'.format(i-9))
        plt.imshow(cat_map_arr[i])
        plt.axis('off')
        #cv2.imwrite('./test/{}.png'.format(str(i)), cat_map_arr[i])
    output_image.savefig('/home/birl/ros/baxter_ws/src/bigjun/dope/src/training/dope_net_output_{}.png'.format(object_name))   

def map_process(feature):
    ''' normalize to [0,1] and then to [0,255] '''
    # use sigmod to [0,1]
    feature= 1.0 / (1 + np.exp(-1 * feature))
    # to [0,255]
    feature=np.round(feature*255, 0) 

    return feature 

