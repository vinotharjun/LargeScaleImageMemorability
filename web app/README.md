# Django and Flask implementation of the Image memorability prediction

### STEPS
* Start the flask API 
```
py deploy.py
```
* can get results from the Terminal
```
#Request 

curl -X POST http:127.0.0.1:5000/predict -F "image=@<image path>"

#Response
{
    MemorabilityScore" : score in range of (0 - 1)
   "heatmap":base64 of the heatmap
 }
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
![](https://github.com/vinotharjun/LargeScaleImageMemorability/blob/master/Photos/Upload%20Image.png)

* Slide the bar to change the values
![](https://github.com/vinotharjun/LargeScaleImageMemorability/blob/master/Photos/slider%20at%20min.png)
![](https://github.com/vinotharjun/LargeScaleImageMemorability/blob/master/Photos/output.png)
![](https://github.com/vinotharjun/LargeScaleImageMemorability/blob/master/Photos/slider%20at%20max.png)

