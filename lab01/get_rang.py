import cv2
import numpy as np

def nothing(x):
    pass

# Carregar a imagem fixa
image_path = 'gamora_nebula.jpg'  # Insira o caminho para sua imagem aqui
frame = cv2.imread(image_path)

# Criar uma janela chamada "Trackbars"
cv2.namedWindow("Trackbars")

# Criar os trackbars para ajustar os valores de intervalo HSV
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    # Convertendo a imagem de BGR para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Obtenção dos valores dos trackbars
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
 
    # Definição do intervalo inferior e superior HSV
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    # Filtragem da imagem para obter a máscara binária
    mask = cv2.inRange(hsv, lower_range, upper_range)
 
    # Aplicação da máscara para visualizar apenas a parte correspondente ao objeto alvo
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Conversão da máscara binária em uma imagem de 3 canais
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Empilhamento da máscara, frame original e resultado filtrado
    stacked = np.hstack((mask_3, frame, res))
    
    # Exibição do frame empilhado com 40% do tamanho original
    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.4, fy=0.4))
    
    # Verificação se o usuário pressionou a tecla ESC para sair do loop
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    # Se o usuário pressionar 's', os valores dos trackbars são impressos e salvos em um arquivo numpy (.npy)
    if key == ord('s'):
        thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
        print(thearray)
        np.save('hsv_value', thearray)
        break

# Fechamento da janela
cv2.destroyAllWindows()
