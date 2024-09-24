import sys

import cv2
import numpy as np

# Carregar uma imagem
image = cv2.imread('../images/museum1.jpg', cv2.IMREAD_GRAYSCALE)


# Verificar se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem!")
    sys.exit(1)

##############################
# SIFT
##############################
# Criar o objeto SIFT
sift = cv2.SIFT_create()

# Detectar key points
keypoints, descriptors = sift.detectAndCompute(image, None)

# Desenhar key points na imagem
img_sift = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


##############################
# ORB
##############################

# Criar o objeto ORB
orb = cv2.ORB_create()

# Detectar key points
keypoints = orb.detect(image, None)

# Calcula descritores ORB
keypoints, descriptors = orb.compute(image, keypoints)

# Desenha keypoints
img_orb = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)


# Mostra as imagens
cv2.imshow('SIFT', img_sift)
cv2.imshow('ORB', img_orb)
cv2.waitKey(0)
cv2.destroyAllWindows()
