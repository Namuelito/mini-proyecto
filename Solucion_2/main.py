import cv2 as cv
import numpy as np
import glob
import os
from red import mynn
import torch
import torchvision.transforms as transforms

classes = ["Alicate", "Destornillador", "Huincha", "Llave"]

transform = transforms.Compose([transforms.ToTensor(),  transforms.Grayscale()])
soft = torch.nn.functional.softmax

def main():
    model = mynn()
    model.load_state_dict(torch.load("pesos.pt"))
    # Directorio de las imágenes
    input_dir = os.path.dirname(__file__) + "/fotos/"

    # Obtiene la lista de todas las imágenes .jpg en el directorio
    imagenes = glob.glob(os.path.join(input_dir, "14.jpg"))     #3 y4 funcan

    for imagen_path in imagenes:
        # Lee la imagen
        A = cv.imread(imagen_path)
        if A is None:
            print(f"No se pudo leer la imagen {imagen_path}")
            continue

        # Obtiene el nombre del archivo sin la ruta y sin la extensión
        nombre_base = os.path.basename(imagen_path)
        nombre_sin_ext = os.path.splitext(nombre_base)[0]

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

        # Filtra los contornos por área mínima
        area_minima = 2000  # Ajustar este valor según sea necesario

        # Después de filtrar los contornos grandes
        contornos_grandes = [contorno for contorno in contornos if cv.contourArea(contorno) > area_minima]

        # Si no hay contornos grandes, continuar
        if not contornos_grandes:
            print(f"No se encontraron contornos grandes en: {imagen_path}")
            continue

        # Iterar y mostrar cada contorno grande
        Encontrados_1 = []
        Encontrados_2 = []

        for i, contorno in enumerate(contornos_grandes):
            img_contorno = np.zeros_like(Agris)
            img_contorno2 = np.zeros_like(Agris)
            cv.drawContours(img_contorno, [contorno], -1, (255), 1)

            # Encontrar el cuadro delimitador del contorno más grande
            x, y, w, h = cv.boundingRect(contorno)

            # # Añadir relleno al cuadro delimitador
            padding = 10
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + 2 * padding, A.shape[1] - x)
            h = min(h + 2 * padding, A.shape[0] - y)

            # Recortar la imagen al nuevo cuadro delimitador
            cropped_image = img_contorno[y:y + h, x:x + w]
            y2,x2 = img_contorno2.shape
            
            diff_x= (x2//2 )- ((x+x+w)//2)
            diff_y= (y2//2 )- ((y+y+h)//2)
            img_contorno2[y+diff_y: y+h + diff_y , x+diff_x: (x+w) + diff_x ] = cropped_image

            y, x = cropped_image.shape[:2]

            margen = (x - y)//2

            if x > y:
                square_image = cv.copyMakeBorder(cropped_image, margen, margen, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
                lab=classes[torch.argmax(soft(model(transform(cv.resize(square_image, (400,400) ))), dim=1)).numpy()]
                Encontrados_1.append(lab)
                # cv.imwrite(f"temp_{nombre_sin_ext}_{i}_{lab}.jpg", square_image)
            else:
                square_image2 = cv.copyMakeBorder(cropped_image, 0, 0, -margen, -margen, cv.BORDER_CONSTANT, value=[0, 0, 0])
                lab=classes[torch.argmax(soft(model(transform(cv.resize(square_image2, (400,400) ))), dim=1)).numpy()]
                Encontrados_2.append(lab)
                # cv.imwrite(f"temp_{nombre_sin_ext}_{i}_{lab}.jpg", square_image2)
        Total = Encontrados_1 + Encontrados_2

        set_classes = set(classes)
        set_total = set(Total)

        # Encontrar todos los elementos distintos entre ambas listas
        distinct_elements = set_classes ^ set_total
        
        if len(distinct_elements) > 0:
            print(f"Las herramientas que faltan en la imagen {nombre_base} son: {distinct_elements}")
        else:
            print(f"En {nombre_base} no faltan herramientas")

if __name__ == "__main__":
    main()

# Resultados:
# Prueba 8: No faltan herramientas. Destornillador confundido por llave.
# Prueba 4: Falta Alicate y destornillador. Destornillador confundido por llave.
# Prueba 7: No faltan herramientas. [Todo bien]
# Prueba 9: No faltan herramientas. [Todo bien]
# Prueba: Falta Llave.              [Todo bien]
# Prueba 2: No faltan herramientas. [Todo bien]
# Prueba 6: No faltan herramientas. [Todo bien]