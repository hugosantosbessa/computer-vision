import cv2
import numpy as np

# Carregar as imagens 
circle = cv2.imread("imagens/circle.jpg", cv2.IMREAD_UNCHANGED)
line = cv2.imread("imagens/line.jpg", cv2.IMREAD_UNCHANGED)

# Criar uma imagem branca
white_bg = np.ones((300, 300, 3), dtype=np.uint8) * 255

# Adicionar a cabeca ao centro da imagem
white_bg[0:100, 100:200] = circle

"""
    FORMANDO O TRONCO
"""
# Dimensaoes
width_line = line.shape[1]
height_line = line.shape[0]

# Rotacao de 90 graus (horizontal para vertical)
x_center = width_line/2
y_center = height_line/2
M_rotation = cv2.getRotationMatrix2D((x_center,y_center),90,1)
tronco = cv2.warpAffine(line,M_rotation,(width_line,height_line))
# Ajustar bordas
tronco = tronco[10:90]
# Adicionar ao back ground
width_tronco = tronco.shape[0]
height_tronco = tronco.shape[1]
white_bg[72:72+width_tronco, 100:100+height_tronco] = tronco

"""
    FORMANDO OS BRACOS
"""
# Diminuir a escala do line para o braco (0.75)
M_scaling = np.float32([[0.75,0,0],[0,1,0]])
braco = cv2.warpAffine(line,M_scaling,(width_line,height_line))
# Ajustar a borda
braco = braco[20:80:,7:67]
# Adicionar ao back ground
white_bg[70:70+braco.shape[0], 87:87+braco.shape[1]] = braco
white_bg[70:70+braco.shape[0], 152:152+braco.shape[1]] = braco

"""
    FORMANDO AS PERNAS
"""
# Redimensionar a perna para ser 2x maior que o braco
perna = cv2.resize(braco, (braco.shape[0]*2, braco.shape[1]), interpolation=cv2.INTER_LINEAR)

# Adicionar borda branca ao altura da imagem
white = (255, 255, 255)
borda_y = 30
perna = cv2.copyMakeBorder(perna, borda_y, borda_y, 0, 0, cv2.BORDER_CONSTANT, value=white)
# Adicionando borda branca a imagem (susbtituir parte preta ao redimensionar para branco)
width_perna = perna.shape[1]
height_perna = perna.shape[0]
border_size = max(width_perna, height_perna)
perna = cv2.copyMakeBorder(perna, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=white)
# Ajustando as dimensões da borda
new_width_perna = perna.shape[1]
new_height_perna = perna.shape[0]
x_center = new_width_perna / 2
y_center = new_height_perna / 2
# Rotação da imagem
M_rotation = cv2.getRotationMatrix2D((x_center, y_center), 135, 1)
perna = cv2.warpAffine(perna, M_rotation, (new_width_perna, new_height_perna))
# Recortar a imagem rotacionada para remover a area extra adicionada pela borda
perna = perna[border_size:new_height_perna-border_size, border_size:new_width_perna-border_size]
# Ajustar as bordas
perna = perna[17:107, 15:105]
# Adicionar a perna ao back ground
width_perna = perna.shape[1]
height_perna = perna.shape[0]
white_bg[150:150+height_perna, 150:150+width_perna] = perna

# Inverter a perna 
perna_invertida = cv2.flip(perna, 1)
# Adicionar a outra perna ao back ground
white_bg[150:150+height_perna, 60:60+width_perna] = perna_invertida

# Mostrar a imagem resultante
cv2.imshow('Result Image', white_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
