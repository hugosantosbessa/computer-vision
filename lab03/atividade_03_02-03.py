# Importar as bibliotecas necessárias novamente, pois o ambiente foi reiniciado
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para mostrar a imagem
def mostrar_imagem(titulo, imagem):
    plt.figure(figsize=(5, 5))
    plt.title(titulo)
    plt.imshow(imagem, cmap='gray')
    plt.axis('off')
    plt.show()

# Importar a imagem fornecida
brasil_image_path = 'imagens/brasil.jpg'
brasil_image = cv2.imread(brasil_image_path)

# Função para converter a imagem para escala de cinza usando a fórmula fornecida
def converter_para_cinza_formula(imagem):
    # Separar os canais de cor
    R = imagem[:, :, 2]  # Canal vermelho
    G = imagem[:, :, 1]  # Canal verde
    B = imagem[:, :, 0]  # Canal azul
    
    # Aplicar a fórmula de conversão para tons de cinza
    Y = 0.3 * R + 0.59 * G + 0.11 * B
    
    # Retornar a imagem em escala de cinza
    return np.uint8(Y)

# Aplicar a conversão para tons de cinza usando a fórmula fornecida
brasil_cinza_formula = converter_para_cinza_formula(brasil_image)

# Mostrar a imagem original e a imagem em tons de cinza
mostrar_imagem("Imagem Original (Brasil)", cv2.cvtColor(brasil_image, cv2.COLOR_BGR2RGB))

# Mostrar a imagem em tons de cinza com a fórmula
mostrar_imagem("Imagem em Tons de Cinza com Fórmula (Brasil)", brasil_cinza_formula)

# Aplicar filtro sépia
def aplicar_filtro_sepia(imagem):
    # Criar matriz de transformação sépia
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    
    # Aplicar o filtro sépia
    sepia_image = cv2.transform(imagem, sepia_filter)
    
    # Garantir que os valores fiquem entre 0 e 255
    sepia_image = np.clip(sepia_image, 0, 255)
    
    return np.uint8(sepia_image)

# Converter a imagem original para sépia
brasil_sepia = aplicar_filtro_sepia(brasil_image)

# Mostrar a imagem em sépia
mostrar_imagem("Imagem com Filtro Sépia (Brasil)", cv2.cvtColor(brasil_sepia, cv2.COLOR_BGR2RGB))
