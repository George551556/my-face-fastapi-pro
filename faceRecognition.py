import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
from PIL import Image
from os import environ
'''该文件用于将输入的图片与库中人脸进行对比，返回相似度最高的人名'''

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"# lkz添加的代码

from nets.facenet import Facenet as facenet
from utils.utils import preprocess_input, resize_image, show_config

# 指定人脸仓库目录路径
dir_path = "img"
threshold = 0.95

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和backbone需要修改！
#--------------------------------------------#
class Facenet(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测要修改model_path，指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表准确度较高，仅代表该权值在验证集上泛化性能较好。
        #--------------------------------------------------------------------------#
        # model_data/facenet_mobilenet.pth
        "model_path"    : "model_data/double-att.pth",
        #--------------------------------------------------------------------------#
        #   输入图片的大小。
        #--------------------------------------------------------------------------#
        "input_shape"   : [160, 160, 3],
        #--------------------------------------------------------------------------#
        #   所使用到的主干特征提取网络
        #--------------------------------------------------------------------------#
        "backbone"      : "mobilenet",
        #-------------------------------------------#
        #   是否进行不失真的resize
        #-------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Facenet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()
        
        show_config(**self._defaults)
        
    def generate(self):
        #---------------------------------------------------#
        #   载入模型与权值
        #---------------------------------------------------#
        #------------------------------------------------------------------#
        #   phiSelect       使用到的注意力类型attention
        #                   0:不使用   1:se_block    2:cbam_block    3:CA_block
        phiSelect           = 2
        #------------------------------------------------------------------#
        print('Loading weights into state dict...')
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net    = facenet(backbone=self.backbone, mode="predict", phi=phiSelect).eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
    
    #---------------------------------------------------#
    #   检测图片：将image_1图片与库中图片比对
    #---------------------------------------------------#
    def detect_image(self, image_1):
        #---------------------------------------------------#
        #   图片预处理，归一化
        #---------------------------------------------------#
        with torch.no_grad():
            image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(preprocess_input(np.array(image_1, np.float32)), (2, 0, 1)), 0))
            if self.cuda:
                photo_1 = photo_1.cuda()
            output1 = self.net(photo_1).cpu().numpy()
            
            # 遍历目录中的文件，找到最符合的图片
            distance = 999.0
            targetName = "无匹配结果"
            for filename in os.listdir(dir_path):
                # 构造图片路径
                img_path = os.path.join(dir_path, filename)
                # 打开图片并进行相关操作
                with Image.open(img_path) as image_2:
                    # 对图片进行操作，例如调整大小、裁剪等
                    image_2 = resize_image(image_2, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
                    photo_2 = torch.from_numpy(np.expand_dims(np.transpose(preprocess_input(np.array(image_2, np.float32)), (2, 0, 1)), 0))
                    if self.cuda:
                        photo_2 = photo_2.cuda()
                    output2 = self.net(photo_2).cpu().numpy()

                    #   计算二者之间的距离
                    l1 = np.linalg.norm(output1 - output2, axis=1)
                    if l1<distance and l1<threshold:
                        distance = l1
                        targetName = filename
            
            if(distance==999.0):
                return targetName, distance
            else:
                targetName = targetName[:-7]
                return targetName, distance


        #     #---------------------------------------------------#
        #     # BUG:解决方案，lkz添加的代码段
        #     #---------------------------------------------------#
        #     def suppress_qt_warnings():
        #         environ["QT_DEVICE_PIXEL_RATIO"] = "0"
        #         environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        #         environ["QT_SCREEN_SCALE_FACTORS"] = "1"
        #         environ["QT_SCALE_FACTOR"] = "1"
        # suppress_qt_warnings()#lkz添加的代码

        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(image_1))
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(image_2))
        # plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va= 'bottom',fontsize=11)
        # plt.show()
        # return l1
