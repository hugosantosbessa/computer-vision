import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage import filters
from skimage import color
import os
from matplotlib import animation

def calculate_energy(image):
    gray_image = color.rgb2gray(image)
    energy = np.abs(filters.sobel_h(gray_image)) + np.abs(filters.sobel_v(gray_image))
    return energy

def find_vertical_seam(energy):
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, r):
        for j in range(c):
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            elif j == c-1:
                idx = np.argmin(M[i-1, j-1:j+1])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]
            M[i, j] += min_energy

    return M, backtrack

def find_horizontal_seam(energy):
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for j in range(1, c):
        for i in range(r):
            if i == 0:
                idx = np.argmin(M[i:i+2, j-1])
                backtrack[i, j] = idx + i
                min_energy = M[idx + i, j-1]
            elif i == r-1:
                idx = np.argmin(M[i-1:i+1, j-1])
                backtrack[i, j] = idx + i - 1
                min_energy = M[idx + i - 1, j-1]
            else:
                idx = np.argmin(M[i-1:i+2, j-1])
                backtrack[i, j] = idx + i - 1
                min_energy = M[idx + i - 1, j-1]
            M[i, j] += min_energy

    return M, backtrack

def remove_vertical_seam(image, backtrack):
    r, c, _ = image.shape
    output = np.zeros((r, c - 1, 3), dtype=image.dtype)
    j = np.argmin(backtrack[-1])
    for i in reversed(range(r)):
        output[i, :, 0] = np.delete(image[i, :, 0], [j])
        output[i, :, 1] = np.delete(image[i, :, 1], [j])
        output[i, :, 2] = np.delete(image[i, :, 2], [j])
        j = backtrack[i, j]
    return output

def remove_horizontal_seam(image, backtrack):
    r, c, _ = image.shape
    output = np.zeros((r - 1, c, 3), dtype=image.dtype)
    i = np.argmin(backtrack[:, -1])
    for j in reversed(range(c)):
        output[:, j, 0] = np.delete(image[:, j, 0], [i])
        output[:, j, 1] = np.delete(image[:, j, 1], [i])
        output[:, j, 2] = np.delete(image[:, j, 2], [i])
        i = backtrack[i, j]
    return output

def visualize_seam_removal(image, backtrack, seam_direction='vertical'):
    seam_image = np.copy(image)
    r, c, _ = image.shape
    if seam_direction == 'vertical':
        j = np.argmin(backtrack[-1])
        for i in reversed(range(r)):
            seam_image[i, j, :] = [255, 0, 0]  # Marcar a costura em vermelho
            j = backtrack[i, j]
    elif seam_direction == 'horizontal':
        i = np.argmin(backtrack[:, -1])
        for j in reversed(range(c)):
            seam_image[i, j, :] = [255, 0, 0]  # Marcar a costura em vermelho
            i = backtrack[i, j]
    return seam_image

def seam_carving_with_animation(image, num_seams, seam_direction='vertical', save_path='seam_carving.gif'):
    frames = []
    for _ in range(num_seams):
        energy = calculate_energy(image)
        if seam_direction == 'vertical':
            M, backtrack = find_vertical_seam(energy)
            seam_image = visualize_seam_removal(image, backtrack, 'vertical')
            image = remove_vertical_seam(image, backtrack)
        elif seam_direction == 'horizontal':
            M, backtrack = find_horizontal_seam(energy)
            seam_image = visualize_seam_removal(image, backtrack, 'horizontal')
            image = remove_horizontal_seam(image, backtrack)
        frames.append(seam_image)

    # Criar a animação e salvar como GIF
    fig, ax = plt.subplots()
    def update(frame):
        ax.imshow(frame)
        ax.axis('off')
    ani = animation.FuncAnimation(fig, update, frames=frames, repeat_delay=500)
    ani.save(save_path, writer='imagemagick', fps=5)
    plt.close()
    return image  # Retorna a imagem final após o seam carving

# Carregar a imagem
img = io.imread('balls.jpg')

# Aplicar o seam carving com animação para costuras verticais
carved_image_vertical = seam_carving_with_animation(img.copy(), num_seams=60, seam_direction='vertical', save_path='vertical_seam_carving.gif')

# Aplicar o seam carving com animação para costuras horizontais
carved_image_horizontal = seam_carving_with_animation(img.copy(), num_seams=40, seam_direction='horizontal', save_path='horizontal_seam_carving.gif')

# Mostrar a imagem original e as modificadas (Vertical e Horizontal)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(carved_image_vertical)
ax[1].set_title('Vertical Seam Carved Image')
ax[1].axis('off')

ax[2].imshow(carved_image_horizontal)
ax[2].set_title('Horizontal Seam Carved Image')
ax[2].axis('off')

plt.tight_layout()

# Salvar a imagem final com todas as comparações
plt.savefig('result.png')
plt.show()
