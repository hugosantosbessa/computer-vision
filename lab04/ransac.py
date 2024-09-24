import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_cantos_harris(caminho_imagem):
    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        print("Erro: Imagem não encontrada")
        return
    
    imagem_copia = np.copy(imagem)
    imagem_cinza = cv2.cvtColor(imagem_copia, cv2.COLOR_BGR2GRAY)
    imagem_cinza = np.float32(imagem_cinza)

    # Detectar cantos usando o método Harris
    cantos_harris = cv2.cornerHarris(imagem_cinza, blockSize=2, ksize=3, k=0.01)
    pontos_cantos = np.argwhere(cantos_harris > 0.01 * cantos_harris.max())
    pontos_cantos = np.flip(pontos_cantos, axis=1)

    # Configuração do gráfico
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(imagem_copia, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Função para ajustar a linha usando o algoritmo RANSAC
    def ajustar_linha_ransac(pontos, limite=5.0, iteracoes=50):
        melhor_linha = None
        melhor_inliers = 0

        for _ in range(iteracoes):
            # Selecionar dois pontos aleatórios
            amostra = pontos[np.random.choice(pontos.shape[0], 2, replace=False)]
            (x1, y1), (x2, y2) = amostra

            # Calcular os parâmetros da linha (ax + by + c = 0)
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2

            # Calcular a distância de todos os pontos até a linha
            distancias = np.abs(a * pontos[:, 0] + b * pontos[:, 1] + c) / np.sqrt(a**2 + b**2)
            inliers = pontos[distancias < limite]
            num_inliers = len(inliers)

            # Atualizar a melhor linha se esta for melhor
            if num_inliers > melhor_inliers:
                melhor_inliers = num_inliers
                melhor_linha = (a, b, c)

        # Plotar a melhor linha encontrada
        if melhor_linha is not None:
            a, b, c = melhor_linha
            x_vals = np.array([0, imagem.shape[1]])
            y_vals = -(a * x_vals + c) / b
            ax.plot(x_vals, y_vals, 'g-', linewidth=2)

        plt.show()
        return melhor_linha

    # Ajustar a linha usando o algoritmo RANSAC
    melhor_linha = ajustar_linha_ransac(pontos_cantos)

if __name__ == "__main__":
    detectar_cantos_harris('pontos_ransac.png')
