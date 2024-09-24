import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para leitura e redimensionamento de imagens
def read_and_resize(image_path, scale=0.5):
    img = cv2.imread(image_path)
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

# Função para converter imagem para escala de cinza
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Função para calcular keypoints e descritores usando SIFT
def compute_sift_features(image):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)

# Função para realizar correspondência utilizando o Ratio Test de Lowe
def filter_matches(descriptors1, descriptors2, ratio=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    return [m for m, n in matches if m.distance < ratio * n.distance]

# Função para calcular a homografia e transformar a imagem
def compute_homography_transform(img1, img2, keypoints1, keypoints2, matches, ransac_thresh=5.0):
    if len(matches) < 4:
        raise ValueError("Número insuficiente de keypoints para calcular a homografia.")
    
    # Extração das coordenadas dos keypoints das correspondências válidas
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Calcula a homografia
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    
    # Define os quatro cantos da imagem inicial para projeção
    h, w = img1.shape[:2]
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    # Calcula os limites da nova área que contém as duas imagens
    x_min, y_min = np.int32(transformed_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(transformed_corners.max(axis=0).ravel())
    
    # Ajusta os limites com base no tamanho da segunda imagem
    x_max = max(x_max, img2.shape[1])
    y_max = max(y_max, img2.shape[0])
    
    # Calcula a translação necessária
    translation = [-x_min, -y_min]
    
    # Cria a matriz de translação
    translation_matrix = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
    
    # Ajusta a homografia adicionando a translação
    adjusted_H = translation_matrix.dot(H)
    
    # Define o tamanho do resultado final
    result_size = (x_max - x_min, y_max - y_min)
    
    # Aplica a transformação à primeira imagem
    transformed_img1 = cv2.warpPerspective(img1, adjusted_H, result_size)
    
    # Coloca a segunda imagem na posição correta
    transformed_img1[translation[1]:translation[1] + img2.shape[0], translation[0]:translation[0] + img2.shape[1]] = img2
    
    return transformed_img1

# Função para exibir uma imagem usando Matplotlib
def display_image(image, title="Imagem"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Leitura e redimensionamento das imagens
resized_img1 = read_and_resize("campus_quixada1.png")
resized_img2 = read_and_resize("campus_quixada2.png")

# Conversão para escala de cinza
gray_img1 = convert_to_gray(resized_img1)
gray_img2 = convert_to_gray(resized_img2)

# Cálculo de keypoints e descritores
keypoints1, descriptors1 = compute_sift_features(gray_img1)
keypoints2, descriptors2 = compute_sift_features(gray_img2)

# Filtragem de correspondências válidas
valid_matches = filter_matches(descriptors1, descriptors2)

# Computação da homografia e transformação da imagem
combined_img = compute_homography_transform(resized_img1, resized_img2, keypoints1, keypoints2, valid_matches)

# Exibição das imagens uma por uma
display_image(resized_img1, "Imagem 1 Original")
display_image(resized_img2, "Imagem 2 Original")
display_image(combined_img, "Imagem Combinada")

# Salvamento da imagem combinada em um arquivo
cv2.imwrite("imagem_combinada.png", combined_img)
print("Imagem combinada salva como 'imagem_combinada.png'.")
