#### 简介

使用Bubbling导的人脸识别项目：[源码地址](https://github.com/bubbliiiing/facenet-keras)  [博客地址](https://blog.csdn.net/weixin_44791964/article/details/107513379) 

将原项目添加上attention模块，并用400人的若干戴口罩数据集进行再训练，将预测模型使用fastapi部署

#### 使用说明

在我的电脑上是名为pytorch的虚拟环境，控制台输入以下代码即可

```
uvicorn main:app --reload
```

浏览器 127.0.0.1:8000/docs 可以访问fastapi官方给出的文档界面。

#### 各文件说明

文件夹img：作为人脸数据库，存放录入的人脸图片，命名方式：xxx.01.jpg   , xxx.02.jpg

model_data：存放权重文件

temp_upload: 是项目运行时，暂时存放前端提交的图片用于识别

testimage：一个测试用的

facenet.py: 模型结构

main_test.py: 用于开发测试

main.py: 用fastapi部署的接口