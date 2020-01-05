# Django and Flask implementation of the Image memorability prediction

### STEPS
* Start the flask API 
```
py deploy.py
```
* still u can get results from the Terminal
```
curl -X POST http:127.0.0.1:5000/predict -F "image=@<image path>"
````
* In the Django Project run 
```
pip install requirements.txt
```
* Start the Deployment Server
```
py manage.py runserver
```
* Now in the upload image card select the image for which you need the memorability score and heat map image

* Slide the bar to change the values
#### 0-Original Image
#### 1-Complete Heat Map
