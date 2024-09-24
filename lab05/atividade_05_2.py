import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para calcular a energia da imagem usando gradientes do OpenCV
def calcular_energia(img, mascara):
    imagem_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(imagem_cinza, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(imagem_cinza, cv2.CV_64F, 0, 1, ksize=3)
    energia = np.sqrt(grad_x**2 + grad_y**2)
    energia[mascara == 255] -= 1000
    return energia

# Função para criar uma máscara para o objeto a ser removido
def criar_mascara(img, cor_baixa, cor_alta):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(img_hsv, cor_baixa, cor_alta)
    return mascara

# Função para encontrar a costura com menor energia
def encontrar_costura(energia):
    linhas, colunas = energia.shape
    M = energia.copy()
    caminho = np.zeros_like(M, dtype=int)

    for i in range(1, linhas):
        for j in range(colunas):
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                caminho[i, j] = idx + j
                min_energia = M[i-1, idx + j]
            elif j == colunas - 1:
                idx = np.argmin(M[i-1, j-1:j+1])
                caminho[i, j] = idx + j - 1
                min_energia = M[i-1, idx + j - 1]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                caminho[i, j] = idx + j - 1
                min_energia = M[i-1, idx + j - 1]
            M[i, j] += min_energia

    return M, caminho

# Função para remover a costura da imagem e da máscara
def remover_costura(img, caminho, mascara=None):
    linhas, colunas, _ = img.shape
    saida = np.zeros((linhas, colunas - 1, 3), dtype=img.dtype)
    nova_mascara = np.zeros((linhas, colunas - 1), dtype=mascara.dtype) if mascara is not None else None
    j = np.argmin(caminho[-1])
    
    for i in reversed(range(linhas)):
        saida[i, :, 0] = np.delete(img[i, :, 0], [j])
        saida[i, :, 1] = np.delete(img[i, :, 1], [j])
        saida[i, :, 2] = np.delete(img[i, :, 2], [j])
        if mascara is not None:
            nova_mascara[i, :] = np.delete(mascara[i, :], [j])
        j = caminho[i, j]

    return saida, nova_mascara

# Função principal que aplica o seam carving
def aplicar_seam_carving(img, mascara, num_costuras):
    for _ in range(num_costuras):
        energia = calcular_energia(img, mascara)
        M, caminho = encontrar_costura(energia)
        img, mascara = remover_costura(img, caminho, mascara)
    return img



# Caminho para a imagem de entrada fornecida pelo usuário
caminho_imagem = 'maxresdefault_with_red_ball.jpg'
imagem = cv2.imread(caminho_imagem)
cor_baixa = np.array([0, 100, 100])
cor_alta = np.array([10, 255, 255])
mascara_objeto = criar_mascara(imagem, cor_baixa, cor_alta)

plt.imshow(mascara_objeto, cmap='gray')
plt.title('Máscara para Remoção')
plt.axis('off')
plt.show()


imagem_resultante = aplicar_seam_carving(imagem, mascara_objeto, 100)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(imagem_resultante, cv2.COLOR_BGR2RGB))
plt.title('Imagem Sem Objeto')
plt.axis('off')

plt.show()
