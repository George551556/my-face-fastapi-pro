import numpy as np
import uvicorn
import cv2
import os
from io import BytesIO
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, RedirectResponse
# from fastapi.templating import Jinja2Templates
from PIL import Image
from faceRecognition import Facenet

# uvicorn main:app --reload
app = FastAPI()
model = Facenet()

@app.get('/')
def root():
    return "hello face recognition"

@app.post('/file')
async def face_reco(file: UploadFile = File(...)):
    #保存图片到本地
    filename = file.filename
    path = os.path.join('temp_upload', filename)
    with open(path, 'wb') as f:
        f.write(await file.read())
    # 读取本地图片
    image = Image.open(path)
    result = model.detect_image(image)
    name, distance = result
    # 设置返回信息
    return_result = "name: "+ str(name)+"      " +"distance: "+str(distance)
    # return {"result":'success'}
    return return_result
    return {"result":return_result}


