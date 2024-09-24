import cv2
import numpy as np

def make_image_yellow(img, intensity=0.5):
    """
    Aumenta a intensidade do amarelo na imagem.
    
    Parâmetros:
    img (numpy.ndarray): Imagem de entrada em formato BGR.
    intensity (float): Intensidade do efeito amarelado. Valor entre 0 e 1.
    
    Retorno:
    numpy.ndarray: Imagem com efeito amarelado aplicado.
    """
    # Garante que a intensidade esteja no intervalo [0, 1]
    intensity = np.clip(intensity, 0, 1)
    
    # Copia a imagem original
    yellow_img = img.copy()
    
    # Separa os canais B, G e R
    B, G, R = cv2.split(yellow_img)
    
    # Reduz a intensidade do canal azul (B) e aumenta a do vermelho (R) e verde (G)
    B = B * (1 - intensity)
    G = G + (255 - G) * intensity
    R = R + (255 - R) * intensity
    
    # Combina novamente os canais
    yellow_img = cv2.merge([B.astype(np.uint8), G.astype(np.uint8), R.astype(np.uint8)])
    
    return yellow_img

# Carregar a imagem
image = cv2.imread('imagens/jato.jpg')

# Aplicar o efeito amarelado
yellow_image = make_image_yellow(image, intensity=0.5)

# Mostrar a imagem original e a imagem com efeito
cv2.imshow('Original', image)
cv2.imshow('Amarelada', yellow_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
