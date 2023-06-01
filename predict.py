# Arda Mavi

import sys
from get_dataset import get_img
from keras.models import Sequential
from keras.models import model_from_json
import numpy as np
import cv2
import os
import datetime
import asyncio


def transformPrediction(val):
    return 'cat' if val else 'dog'

def initModel():
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    return model

async def storeImage(dir_path, frame):
    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, "img")
    cv2.imwrite('{}_{}.{}'.format(base_path, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'), "jpg"),frame,  [cv2.IMWRITE_JPEG_QUALITY])
    asyncio.sleep(10)


async def writeFiles(Y, framesList):
    tasks = []
    for i in range(len(Y)):
        prediction = transformPrediction(Y[i])
        task = asyncio.ensure_future(storeImage("output/" + prediction + "s", framesList[i]))
        tasks.append(task)
    await asyncio.gather(*tasks)


def parseVideoFile(img_dir):
    videoFile = cv2.VideoCapture(img_dir)
    framesList = []
    while videoFile.isOpened():
        ret, frame = videoFile.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        framesList.append(frame)
    print(f'Read  {len(framesList)}  frames ')
    videoFile.release()
    cv2.destroyAllWindows()
    return framesList

def predictVideo(img_dir):
    framesList = parseVideoFile(img_dir)
    model = initModel()

    X = np.zeros((len(framesList), 64, 64, 3), dtype='float64')
    index = 0
    for frame in framesList:
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        X[index] = frame
        index = index + 1

    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(writeFiles(Y,framesList))
    loop.close()

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    return transformPrediction(Y[0])

def predictImages(img_dir):
    img = get_img(img_dir)
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    Y = predict(model, X)
    print('It is a ' + Y + ' !')


if __name__ == '__main__':
    img_dir = sys.argv[1]
    if(img_dir.endswith("mp4")):
        predictVideo(img_dir)
    else:
        predictImages(img_dir)


