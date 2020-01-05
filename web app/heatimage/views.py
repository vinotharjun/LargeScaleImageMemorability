from django.shortcuts import render
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from PIL import Image
from io import StringIO
import requests,json
import os
import glob
def index(request):
    return render(request,'index.html')


def imageupload(request): 

    if request.method == 'POST':
        image = request.FILES['images']
        title = image.name
        folder_path = ''+str(title)
        print()
        if not os.path.exists('./media/'+title):
            path = default_storage.save(folder_path, ContentFile(image.read()))
        with open("./media/"+title,"rb") as f:
            data = f.read()
        files = {
        'userid': (None, 'vinoth'),   
        'image': data
        }
        response = requests.post('http://127.0.0.1:500/predict', files=files)
        # response = requests.post('http://a427ae91.ngrok.io/predict', files=files)
        api_values = response.json()
        heatmap_response = api_values['heatmap']
        score = float(api_values['score'])*100
        score = str(int((score*100)+0.5)/100)+'%'

        image = {'content':[{'id':'normal','url':"http://127.0.0.1:8000/media/"+title},{'id':'heatmap','url':"http://127.0.0.1:8000/media/heatmap.jpg"},{'id':"Memorability Score",'Score':score}]}
        temp = {}
        temp['images']=image
        imgdata = base64.b64decode(heatmap_response)
        with open('./media/heatmap.jpg',"wb+") as f:
            f.write(imgdata)

        return render(request,'slider.html',{'images':temp})
    return render(request,'index.html')