import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Abre a imagem
filename = sys.argv[1]
img = cv2.imread(filename)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#outros modelos em:
# https://github.com/opencv/opencv/tree/master/data/haarcascades


## Carrega o modelo
model_path = "./models/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(model_path)

#Detecção
results = detector.detectMultiScale(img_gray, scaleFactor=1.15,minNeighbors=5,minSize=(50, 50))

# Exibe os resultados
for (x,y,w,h) in results:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
  
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()





