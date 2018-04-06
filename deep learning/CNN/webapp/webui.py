import web.webapi
import requests
import shutil
from PIL import Image
import io

import config.settings 
view = config.settings.render

class ImageNet():

    def GET(self):
        web.webapi.header("Content-type","text/html")

        url = web.webapi.input().get('url',"")
        if not url:
            return view.searchimage("this is a simple image classifier")

        response = requests.get(url, stream=True)
        with open("/tmp/imagesample.jpg", "wb") as out:
            file  = io.BytesIO(response.content)
            Image.open(file).save("/tmp/imagesample.jpg", "JPEG")
#            shutil.copyfileobj(response.raw, out)

        import os
        response = os.popen("python /opt/caffe/python/imagenet/app.py").read()
        return view.searchimage(response)
