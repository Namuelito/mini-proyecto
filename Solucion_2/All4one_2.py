import cv2 as cv
import numpy as np
import glob
import os
import random
import shutil

# Directorio de las imágenes
input_dir = os.path.dirname(__file__) + "/Herramientas_raw/"
output_folder = os.path.dirname(__file__) + "/dataset_test/"

# Directorio para las carpetas de entrenamiento y validación
train_dir = os.path.join(output_folder, "train")
val_dir = os.path.join(output_folder, "val")

# Crear las carpetas si no existen
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Obtiene la lista de todas las imágenes .jpg en el directorio
imagenes = glob.glob(os.path.join(input_dir, "*.jpg"))

# Lista para almacenar archivos generados
generated_files = []

for imagen_path in imagenes:
    # Lee la imagen
    A = cv.imread(imagen_path)
    if A is None:
        print(f"No se pudo leer la imagen {imagen_path}")
        continue

    print(f"Trabajando en: {imagen_path}")

    # Transforma a escala de grises
    Agris = cv.cvtColor(A, cv.COLOR_BGR2GRAY)

    # Aplica filtro de desenfoque para reducir el ruido
    Agris_blur = cv.GaussianBlur(Agris, (5, 5), 0)

    # Detección de bordes usando el operador de Sobel
    grad_x = cv.Sobel(Agris_blur, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(Agris_blur, cv.CV_64F, 0, 1, ksize=3)
    grad = cv.magnitude(grad_x, grad_y)
    _, sobel_edges = cv.threshold(grad, 50, 255, cv.THRESH_BINARY)
    sobel_edges = np.uint8(sobel_edges)

    # Encuentra contornos utilizando el método de código cadena
    contornos, _ = cv.findContours(sobel_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Si no hay contornos, continuar
    if not contornos:
        print(f"No se encontraron contornos en: {imagen_path}")
        continue

    # Encuentra el contorno con el área más grande
    contorno_mas_grande = max(contornos, key=cv.contourArea)

    # Crea una imagen en blanco para dibujar el contorno
    img_contorno = np.zeros_like(Agris)

    # Dibuja solo el contorno más grande
    cv.drawContours(img_contorno, [contorno_mas_grande], -1, (255), 1)

    # Obtiene el nombre del archivo sin la ruta y sin la extensión
    nombre_base = os.path.basename(imagen_path)
    nombre_sin_ext = os.path.splitext(nombre_base)[0]

    # Encuentra el cuadro delimitador del contorno más grande
    x, y, w, h = cv.boundingRect(contorno_mas_grande)

    # Agrega padding al cuadro delimitador
    padding = 10
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, A.shape[1] - x)
    h = min(h + 2 * padding, A.shape[0] - y)

    # Recorta la imagen al nuevo cuadro delimitador
    cropped_image = img_contorno[y:y + h, x:x + w]

    y, x = cropped_image.shape[:2]

    margen = (x - y) // 2

    if x > y:
        square_image = cv.copyMakeBorder(cropped_image, margen, margen, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        square_image = cv.copyMakeBorder(cropped_image, 0, 0, -margen, -margen, cv.BORDER_CONSTANT, value=[0, 0, 0])

    height, width = square_image.shape[:2]

    # Rotar y guardar la imagen redimensionada
    for angle in range(360):
        M = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated = cv.warpAffine(square_image, M, (width, height))

        rotated_output_path = os.path.join(output_folder, f"{nombre_sin_ext}_{angle}.jpg")
        cv.imwrite(rotated_output_path, rotated)

        # Agregar el archivo generado a la lista
        generated_files.append(rotated_output_path)

cv.destroyAllWindows()

# Barajar los archivos generados
random.shuffle(generated_files)

# Calcular el número de imágenes por conjunto de entrenamiento y validación
num_images_per_set = len(imagenes)
num_train_images = int(num_images_per_set * 0.8)
num_val_images = num_images_per_set - num_train_images

# Distribuir las imágenes originales de manera equitativa
for imagen_path in imagenes:
    base_name = os.path.splitext(os.path.basename(imagen_path))[0]
    rotated_images = [file for file in generated_files if base_name in file]

    # Determinar si las imágenes rotadas deben ir al conjunto de entrenamiento o validación
    if len(rotated_images) % 5 == 0:
        # 80% a entrenamiento, 20% a validación
        num_train = int(len(rotated_images) * 0.8)
        num_val = len(rotated_images) - num_train
    else:
        # Distribuir equitativamente
        num_train = len(rotated_images) // 5 * 4
        num_val = len(rotated_images) - num_train

    # Mover las imágenes a las carpetas correspondientes
    train_images = rotated_images[:num_train]
    val_images = rotated_images[num_train:]

    for file_path in train_images:
        shutil.move(file_path, os.path.join(train_dir, os.path.basename(file_path)))

    for file_path in val_images:
        shutil.move(file_path, os.path.join(val_dir, os.path.basename(file_path)))

print("División en carpetas de entrenamiento y validación completada.")