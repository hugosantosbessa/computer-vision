import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def match_features(im1, im2, method='surf'):
    """
    Encontra e desenha correspondências entre duas imagens usando SURF ou ORB.

    :param im1: Primeira imagem em escala de cinza.
    :param im2: Segunda imagem em escala de cinza.
    :param method: Método de detecção ('surf' ou 'orb').
    :return: Imagem com correspondências desenhadas.
    """
    if method == 'surf':
        min_hessian = 400
        detector = cv.xfeatures2d_SURF.create(min_hessian)
        norm_type = cv.NORM_L2
        matcher = cv.BFMatcher(norm_type)
        knn_match = True
        distance_ratio = 0.75
    elif method == 'orb':
        detector = cv.ORB_create()
        norm_type = cv.NORM_HAMMING
        matcher = cv.BFMatcher(norm_type, crossCheck=True)
        knn_match = False
        distance_ratio = 1.0

    # Detecta keypoints e descritores
    kp1, des1 = detector.detectAndCompute(im1, None)
    kp2, des2 = detector.detectAndCompute(im2, None)

    # Verifica se os descritores foram encontrados
    if des1 is None or des2 is None:
        print(f"Sem correspondências encontradas com o método {method.upper()}.")
        return np.zeros_like(im1)  # Retorna imagem em preto caso não haja correspondências

    # Encontra correspondências
    matches = matcher.knnMatch(des1, des2, k=2) if knn_match else matcher.match(des1, des2)

    # Filtra correspondências com base na distância
    good_matches = []
    if knn_match:
        for m, n in matches:
            if m.distance < distance_ratio * n.distance:
                good_matches.append([m])
    else:
        good_matches = sorted(matches, key=lambda x: x.distance)[:15]

    # Desenha as correspondências
    img_matches = cv.drawMatchesKnn(im1, kp1, im2, kp2, good_matches, None,
                                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) if knn_match else \
        cv.drawMatches(im1, kp1, im2, kp2, good_matches, None,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print(f"({method.upper()}) Número de correspondências boas: {len(good_matches)}")
    return img_matches


def main(img1_path, img2_path):
    """
    Carrega as imagens, aplica a correspondência de características e exibe os resultados.

    :param img1_path: Caminho para a primeira imagem.
    :param img2_path: Caminho para a segunda imagem.
    """
    # Carrega as imagens em escala de cinza
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Erro ao carregar as imagens. Verifique os caminhos fornecidos.")
        return

    # Encontra e desenha correspondências usando SURF e ORB
    img_match_surf = match_features(img1, img2, method='surf')
    img_match_orb = match_features(img1, img2, method='orb')

    # Exibe as imagens correspondentes lado a lado
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(img_match_surf)
    plt.title('SURF')
    plt.subplot(122)
    plt.imshow(img_match_orb)
    plt.title('ORB')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python script.py <caminho_imagem1> <caminho_imagem2>")
    else:
        main(sys.argv[1], sys.argv[2])
