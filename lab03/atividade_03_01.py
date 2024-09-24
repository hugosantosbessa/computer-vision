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

# Função para aplicar filtros no domínio do espaço e da frequência
def aplicar_filtros(imagem_path, filtro_espaco, raio_filtro_frequencia):
    # Carregar a imagem em escala de cinza
    imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
    
    # Aplicar filtro de média (domínio do espaço)
    imagem_filtro_espaco = cv2.blur(imagem, filtro_espaco)
    
    # Transformada de Fourier para o domínio da frequência
    dft = cv2.dft(np.float32(imagem), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Criar um filtro passa-baixas
    rows, cols = imagem.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= raio_filtro_frequencia**2
    mask[mask_area] = 1
    
    # Aplicar a máscara e a transformada inversa de Fourier
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalizar para exibir
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    return imagem, imagem_filtro_espaco, img_back

# Caminhos para as imagens
imagens_paths = [
    'imagens/salt_noise.png',
    'imagens/halftone.png',
    'imagens/pieces.png'
]

# Aplicar os filtros em todas as imagens
for imagem_path in imagens_paths:
    original, filtro_espaco, filtro_freq = aplicar_filtros(imagem_path, (5, 5), 30)
    
    # Mostrar resultados para cada imagem
    mostrar_imagem(f"Imagem Original ({imagem_path})", original)
    mostrar_imagem(f"Imagem com Filtro de Média (5x5) ({imagem_path})", filtro_espaco)
    mostrar_imagem(f"Imagem com Filtro Passa-Baixas (Domínio da Frequência) ({imagem_path})", filtro_freq)