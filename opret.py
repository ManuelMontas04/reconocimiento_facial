import cv2
import os
import imutils
import numpy as np

# Función para capturar rostros y guardarlos en una carpeta
def capturar_rostros(personName, dataPath, max_count=30, video_source=1):
    """
    Captura rostros de un video y los guarda en la ruta especificada.

    Parámetros:
    - personName: Nombre de la persona (se usará para crear una carpeta con su nombre).
    - dataPath: Ruta base donde se almacenarán las imágenes.
    - max_count: Número máximo de imágenes a capturar (por defecto 30).
    - video_source: Fuente de video (0 para cámara predeterminada, 1 para cámara externa, o ruta de un archivo de video).
    """
    personPath = os.path.join(dataPath, personName)

    if not os.path.exists(personPath):
        print('Carpeta creada:', personPath)
        os.makedirs(personPath)

    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
            count += 1

        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27 or count >= max_count:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Proceso completado. Se capturaron {count} imágenes.")


# Función para entrenar el modelo de reconocimiento facial
def entrenar_modelo(dataPath):
    """
    Entrena un modelo de reconocimiento facial usando las imágenes almacenadas en dataPath.

    Parámetros:
    - dataPath: Ruta base donde se almacenaron las imágenes.
    """
    peopleList = os.listdir(dataPath)
    print('Lista de personas:', peopleList)

    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        print('Leyendo las imágenes de:', nameDir)

        for fileName in os.listdir(personPath):
            print('Rostro:', fileName)
            labels.append(label)
            facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))
        label += 1

    # Entrenar el modelo LBPH
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Entrenando el modelo...")
    face_recognizer.train(facesData, np.array(labels))

    # Guardar el modelo
    face_recognizer.write('modeloLBPHFace.xml')
    print("Modelo almacenado como 'modeloLBPHFace.xml'.")


# Función para reconocer rostros en tiempo real
def reconocer_rostros(dataPath, video_source=1):
    """
    Reconoce rostros en tiempo real usando el modelo entrenado.

    Parámetros:
    - dataPath: Ruta base donde se almacenaron las imágenes.
    - video_source: Fuente de video (0 para cámara predeterminada, 1 para cámara externa, o ruta de un archivo de video).

    Retorna:
    - reconocido: True si se reconoció al menos un rostro, False en caso contrario.
    """
    imagePaths = os.listdir(dataPath)
    print('Personas registradas:', imagePaths)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modeloLBPHFace.xml')

    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    reconocido = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            if result[1] < 70:
                cv2.putText(frame, f'{imagePaths[result[0]]}', (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                reconocido = True  # Se reconoció al menos un rostro
            else:
                cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return reconocido


# Ejecución secuencial de las funciones

if __name__ == "__main__":
    dataPath = r'C:\Users\Manuel\OneDrive\Desktop\entrenar_ia\data'  # Cambia a tu ruta
    personName = 'Guardado'  # Nombre de la persona a capturar

    # Paso 1: Intentar reconocer rostros
    print("Intentando reconocer rostros...")
    reconocido = reconocer_rostros(dataPath)

    if not reconocido:
        print("No se reconoció ningún rostro. Iniciando captura de nuevas imágenes...")
        # Paso 2: Capturar rostros
        capturar_rostros(personName, dataPath)

        # Paso 3: Entrenar el modelo con las nuevas imágenes
        print("Entrenando el modelo con las nuevas imágenes...")
        entrenar_modelo(dataPath)

        # Paso 4: Intentar reconocer rostros nuevamente
        print("Intentando reconocer rostros nuevamente...")
        reconocido = reconocer_rostros(dataPath)

        if reconocido:
            print("Rostro reconocido después del entrenamiento.")
        else:
            print("Aún no se reconoce el rostro.")
    else:
        print("Rostro reconocido. No es necesario capturar nuevas imágenes.")
