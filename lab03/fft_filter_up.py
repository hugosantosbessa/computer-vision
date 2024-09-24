import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

def create_adaptive_filter_mask(img_shape, mask):
    l, c = img_shape

    # Tamanho da máscara desejada (em relação ao tamanho da imagem)
    mask_size = min(l, c) // 20  # Ajuste esse valor conforme necessário

    # Certifique-se de que o tamanho seja ímpar para fácil centralização
    if mask_size % 2 == 0:
        mask_size += 1

    # Cria uma máscara do tamanho total da imagem preenchida com zeros
    filter_mask = np.zeros((l, c), np.float32)
    
    # Redimensiona a máscara original para o tamanho calculado
    mask_resized = cv2.resize(mask, (mask_size, mask_size))

    # Posiciona a máscara redimensionada no centro da imagem
    center_l = l // 2
    center_c = c // 2
    half_mask_size = mask_size // 2

    filter_mask[center_l - half_mask_size:center_l + half_mask_size + 1, 
                center_c - half_mask_size:center_c + half_mask_size + 1] = mask_resized
    
    return filter_mask

# Carrega a imagem a partir do argumento fornecido
filename = sys.argv[1]
img = cv2.imread(filename, 0)

# Dimensões da imagem
l, c = img.shape

# FFT usando OpenCV e shift com numpy
# Com o método cv2.dft(...) o resultado é um array com dois canais, 
# sendo um array para o eixo real e outro para o imaginário
# A imagem precisa ser float
img_fft = np.fft.fftshift(cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT))

# Máscaras no domínio do espaço (escolher uma)

# Média
# mask = np.float32([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * (1 / 9)

# Sobel na direção x
mask = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# Sobel na direção y
# mask = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Laplaciano
# mask = np.float32([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Cria a máscara adaptada ao tamanho da imagem
filter_mask = create_adaptive_filter_mask(img.shape, mask)

# Máscara no domínio da frequência (FFT e shift)
fft_mask = np.fft.fftshift(cv2.dft(np.float32(filter_mask), flags=cv2.DFT_COMPLEX_OUTPUT))
mask_abs = cv2.magnitude(fft_mask[:, :, 0], fft_mask[:, :, 1])

# Normaliza valores para o intervalo min-max (opcional)
# min=0.0
# max=1.0
# mask_abs = cv2.normalize(mask_abs, None, min, max, cv2.NORM_MINMAX)

# Replica a magnitude da máscara para os dois canais
fft_mask[:, :, 0] = mask_abs
fft_mask[:, :, 1] = mask_abs

# Filtra no domínio da frequência
fft_filtered = cv2.multiply(img_fft, fft_mask)

# Inverte a FFT
img_back = cv2.idft(np.fft.ifftshift(fft_filtered))
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Plotagem das imagens
imagens = [img,
           np.log(cv2.magnitude(img_fft[:, :, 0], img_fft[:, :, 1]) + 1),  # Adicionei +1 para evitar log(0)
           mask_abs,
           img_back]

titles = ['Original', 'FFT Original', 'FFT Máscara', 'Filtrada']
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(imagens[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.show()
