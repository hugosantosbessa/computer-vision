import cv2
import numpy as np

def swap_colors(image):
    # Convertendo a imagem para HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definindo os intervalos de cor para azul
    # [[77, 45, 26], [108, 255, 255]]
    lower_blue = np.array([77, 45, 26])
    upper_blue = np.array([108, 255, 255])

    # Criando uma mascara para a cor azul
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Definindo os intervalos de cor para verde
    # [[20, 25, 53], [75, 255, 255]]
    lower_green = np.array([20, 25, 53])
    upper_green = np.array([75, 255, 255])

    # Criando uma mascara para a cor verde
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Substituindo a cor azul por um tom de verde
    hsv_image[mask_blue > 0, 0] = 47  # Altera apenas o componente H para um tom de verde

    # Substituindo a cor verde por um tom de azul
    hsv_image[mask_green > 0, 0] = 109  # Altera apenas o componente H para um tom de azul

    # Convertendo de volta para BGR
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    return result_image

# Carregando a imagem
image = cv2.imread('gamora_nebula.jpg')

# Aplicando a troca de cores
swapped_image = swap_colors(image)

# Exibindo a imagem original e resultante
cv2.imshow('Original', image)
cv2.imshow('Swapped Colors', swapped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
