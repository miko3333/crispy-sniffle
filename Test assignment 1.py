import requests
import shutil
from pymongo import MongoClient
from os import path, makedirs

#соаздаём папки
def createFolder(directory):
    try:
        if not path.exists(directory):
            makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

createFolder('./data/')
createFolder('./data/train/')
createFolder('./data/test/')

#wnids кошек и собак
wnids = [['dogs', 'n02084071'], ['cats', 'n02121808']]

#настрйоки MongoDB
client = MongoClient()
db = client.default

#массив с ссылками
list_of_urls = []


for i in range(len(wnids)):
    #создаём папки для всех wnid
    createFolder('./data/train/{}'.format(wnids[i][0]))
    createFolder('./data/test/{}'.format(wnids[i][0]))
    #получаем urls одним списком и качаем их
    list_of_urls.append(requests.get('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}'.format(wnids[i][1], )).text.encode('utf-8'))
    list_of_urls[i] = list_of_urls[i].decode().split('\r\n')
    #ставим range(len(lines[i])) для скачивания всех пикч
    for j in range(10): #кол-во файлов для каждой категории
        r = requests.get(list_of_urls[i][j], stream=True)
        if r.status_code == 200:
            if j % 5 == 0: #качаем и кидаем файлы в папки 4:1
                with open('data/test/{}/{}.jpg'.format(wnids[i][0], str(i)+str(j)), 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
            else:
                with open('data/train/{}/{}.jpg'.format(wnids[i][0], str(i)+str(j)), 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
        image = {
            'category': wnids[i][0],
            'path': 'data/test/{}/{}.jpg'.format(wnids[i][0], str(i)+str(j))
        }
        images = db.images
        image_id = images.insert_one(image).inserted_id
