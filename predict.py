from PIL import Image
from facenet import Facenet

if __name__ == "__main__":
    model = Facenet()
        
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        
        probability = model.detect_image(image_1,image_2)
        print(probability)


'''
复制下面这段到输入框即可测试
普通
img\pyy_0.jpg
img\pyy_2.jpg

img\1_001.jpg
img\pyy_0.jpg

口罩
img\huozun_musk_1.jpg
img\huozun_musk_2.jpg

img\shuqi_musk.jpg
img\shuqi_unmasked.jpg

img\zyl_musk_1.jpg
img\zyl_musk_2.jpg

'''