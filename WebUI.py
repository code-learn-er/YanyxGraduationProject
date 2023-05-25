from flask import Flask, request, make_response, render_template, jsonify
from flask_cors import CORS
from test import *
import base64
import os,io,cv2
from PIL import Image
import numpy as np
app = Flask(__name__)
CORS(app, supports_credentials=True)

size=(128,128)
def filestorage2img(filestoragelist):  # 从缓冲区列表中读取文件并转化为图片
    images = [] 
    for filestorage in filestoragelist:
        images.append(Image.open(io.BytesIO(filestorage.read())))
    return images
def array2base64str(img): # 将array转化为base64str序列
    img=Image.fromarray(img)
    buf=io.BytesIO()
    img.save(buf,format='png')
    img=base64.b64encode(buf.getbuffer()).decode("ascii")
    return img

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    sx_files = request.files.getlist("sx[]")
    mx_files = request.files.getlist("mx[]")
    sx_img = filestorage2img(sx_files)
    mx_img = filestorage2img(mx_files)

    # patient_path = r"patient/Feiyinxing/303"

    # images = [Image.open(os.path.join(patient_path + ".jpg"))] + [
    #     Image.open(os.path.join(patient_path, i)) for i in os.listdir(patient_path)
    # ]
    # sx_img,mx_img=images[0:1],images[1:]

    results, images, label = deal(sx_img + mx_img, size=size)
    # results=[[cv2.resize(np.array(i), dsize=size, interpolation=cv2.INTER_CUBIC) for j in range(10)] for i in sx_img+mx_img]
    # images=[cv2.resize(np.array(i), dsize=size, interpolation=cv2.INTER_CUBIC) for i in sx_img+mx_img]
   
    for i in range(len(results)):
        for j in range(len(results[i])):
            results[i][j]=array2base64str(results[i][j])
    
    images=[array2base64str(img) for img in images]
    return jsonify({'results': results,'images':images,'label':label})

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=7777)
