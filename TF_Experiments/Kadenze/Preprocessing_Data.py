import os
import urllib.request
import ssl

ssl._create_default_https_context=ssl._create_unverified_context

os.mkdir('img_align_celeba')

for img_i in range(1,11):
    f='000%03d.jpg'%img_i
    url='https://s3.amazonaws.com/cadl/celeb-align/' + f
    print(url,end='\r')
    urllib.request.urlretrieve(url,os.path.join('img_align_celeba',f))

