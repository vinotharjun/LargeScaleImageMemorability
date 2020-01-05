from flask import Flask,jsonify,request
from interface import *
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from model import *
import cv2

app = Flask(__name__)
model  = load_model()
finalconv_name="layer4"
params = list(model.parameters())
linear_layer_weights = np.squeeze(params[-3].cpu().data.numpy())
del params

def hook_feature(module, input, output):
    global feature
    feature = output.cpu().data.numpy()
model._modules.get(finalconv_name).register_forward_hook(hook_feature)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route("/test",methods=["POST"])
def test():
    if request.method =="POST":
        file = request.files["image"]
        print(type(file))
        image_bytes = file.read()
        output = prediction(image_bytes)
        return{
            "memscore":output
        }
@app.route("/predict",methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error":"image file is required"
        })
    
    #print(request.files.get("image"))
    try:
        filestr = request.files['image'].read()
        npimg = np.fromstring(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        width,height,_ = img.shape
        img_pil = Image.fromarray(img)
        output,image= prediction(img_pil)
        activation_map = regression_activation_mapping(image,feature,linear_layer_weights,(height,width))
        map_pil = Image.fromarray(activation_map)
        map_pil.save("./heatmap.jpg")
        buffered = BytesIO()
        map_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8") 
        # print(img_str) 
        return jsonify({
            "score":str(output),
            "heatmap":str(img_str)
        })
    except Exception as e:
        print("error in deploy.py line 63",str(e))
        return jsonify({
            "error":"something went wrong with the uploaded file"
        })
        
def prediction(image):
    tensor = transform_image(image=image)
    output = model(tensor)
    return (output.item(),tensor)

if __name__ == '__main__':
    app.run(threaded=False,port=500)