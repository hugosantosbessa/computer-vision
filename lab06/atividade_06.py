
import sys
import cv2
import numpy as np

# Carregar uma imagem
image = cv2.imread('museum1.jpg', cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem!")
    sys.exit(1)

##############################
# SIFT
##############################
# Criar o objeto SIFT
sift = cv2.SIFT_create()

# Detectar key points e calcular descritores
keypoints_sift, descriptors_sift = sift.detectAndCompute(image, None)

# Desenhar key points na imagem
img_sift = cv2.drawKeypoints(image, keypoints_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


##############################
# ORB
##############################
# Criar o objeto ORB
orb = cv2.ORB_create()

# Detectar key points
keypoints_orb = orb.detect(image, None)

# Calcular descritores ORB
keypoints_orb, descriptors_orb = orb.compute(image, keypoints_orb)

# Desenhar key points na imagem
img_orb = cv2.drawKeypoints(image, keypoints_orb, None, color=(0, 255, 0), flags=0)


# Mostrar as imagens
cv2.imshow('SIFT', img_sift)
cv2.imshow('ORB', img_orb)

# Esperar para fechar as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
