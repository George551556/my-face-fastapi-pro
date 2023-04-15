from PIL import Image
from faceRecognition import Facenet
model = Facenet()

while True:
    image = input('Input image filename:')
    try:
        image = Image.open(image)
    except:
        print('Image Open Error! Try again!')
        continue
    model.detect_image(image)
    print('\n')
    


