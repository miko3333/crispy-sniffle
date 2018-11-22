import requests
import os
import urllib.request
import time

#wnids кошек и собак
wnids = [['dogs', 'n02084071'], ['cats', 'n02121808']]
list_of_urls = []
#получаем urls одним списком
for i in range(len(wnids)):
   list_of_urls.append(requests.get('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}'.format(wnids[i][1], )).text.encode('utf-8'))

#создаёт файлы с urls
for i in range(len(wnids)):
    temp = open('{}_urls_file'.format(wnids[i][0]), 'wb')
    temp.write(list_of_urls[i])

#создаём папки
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

createFolder('./data/')
createFolder('./data/train/')
createFolder('./data/test/')

for i in range(len(wnids)):
    createFolder('./data/train/{}'.format(wnids[i][0]))
    createFolder('./data/test/{}'.format(wnids[i][0]))

#список urls
lines = []
for i in range(len(wnids)):
    lines.append([line.rstrip('\n') for line in open('{}_urls_file'.format(wnids[i][0]))])

#качаем и кидаем файлы в папки 4:1
#ставим range(len(lines[i])) для скачивания всех пикч
from pymongo import MongoClient

client = MongoClient()
db = client.default

for i in range(len(wnids)):
    for j in range(10):
        if j % 5 == 0:
            try:
                url = lines[i][j]
                urllib.request.urlretrieve(lines[i][j], 'data/test/{}/{}.jpg'.format(wnids[i][0], str(i)+str(j)))
                time.sleep(3)
                image = {
                'category': wnids[i][0],
                'path': 'data/test/{}/{}.jpg'.format(wnids[i][0], str(i)+str(j))
                }
                images = db.images
                image_id = images.insert_one(image).inserted_id
            except urllib.error.HTTPError:
                time.sleep(3)
                pass
            except urllib.error.URLError:
                time.sleep(3)
                pass
        else:
            try:
                url = lines[i][j]
                urllib.request.urlretrieve(lines[i][j], 'data/train/{}/{}.jpg'.format(wnids[i][0], str(i)+str(j)))
                time.sleep(3)
                image = {
                'category': wnids[i][0],
                'path': 'data/train/{}/{}.jpg'.format(wnids[i][0], str(i)+str(j))
                }
                images = db.images
                image_id = images.insert_one(image).inserted_id
            except urllib.error.HTTPError:
                time.sleep(3)
                pass
            except urllib.error.URLError:
                time.sleep(3)
                pass

import pprint
for image in images.find():
     pprint.pprint(image)
