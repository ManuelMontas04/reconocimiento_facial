import cv2
import os
import imutils
import numpy as np

def capturar_rostros(personName, dataPath, max_count=300, video_source=0):
    personPath = os.path.join(dataPath, personName)

    if not os.path.exists(personPath):
        print(f'📂 Carpeta creada: {personPath}')
        os.makedirs(personPath)

    # Intentar abrir la cámara con diferentes backends
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_VFW]
    cap = None

    for backend in backends:
        cap = cv2.VideoCapture(video_source, backend)
        if cap.isOpened():
            print(f"✅ Cámara abierta con backend: {backend}")
            break

    if not cap or not cap.isOpened():
        raise Exception(f"❌ No se pudo acceder a la cámara en el índice {video_source}.")

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo capturar el frame.")
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            print("❌ No se detectaron rostros.")

        for (x, y, w, h) in faces:
            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
                count += 1

        cv2.imshow('📸 Capturando Rostros', frame)

        if cv2.waitKey(1) == 27 or count >= max_count:  # Salir con ESC o cuando se alcance el máximo
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Captura completada: {count} imágenes guardadas.")


def entrenar_modelo(dataPath):
    peopleList = os.listdir(dataPath)

    if not peopleList:
        raise Exception("❌ Error: No hay carpetas con imágenes para entrenar.")

    print('👤 Personas registradas:', peopleList)

    labels = []
    facesData = []

    for label, nameDir in enumerate(peopleList):
        personPath = os.path.join(dataPath, nameDir)
        for fileName in os.listdir(personPath):
            img = cv2.imread(os.path.join(personPath, fileName), 0)
            if img is not None:
                facesData.append(img)
                labels.append(label)

    if not facesData:
        raise Exception("❌ Error: No se encontraron imágenes válidas para entrenar.")

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("🧠 Entrenando el modelo...")
    face_recognizer.train(facesData, np.array(labels))
    face_recognizer.write('modeloLBPHFace.xml')
    print("✅ Modelo almacenado como 'modeloLBPHFace.xml'.")


def reconocer_rostros(dataPath, video_source=0, threshold=50):
    if not os.path.isfile('modeloLBPHFace.xml'):
        raise Exception("❌ Error: No se encontró el modelo entrenado 'modeloLBPHFace.xml'.")

    imagePaths = os.listdir(dataPath)
    print('👤 Personas registradas:', imagePaths)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modeloLBPHFace.xml')

    # Intentar abrir la cámara con diferentes backends
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_VFW]
    cap = None

    for backend in backends:
        cap = cv2.VideoCapture(video_source, backend)
        if cap.isOpened():
            print(f"✅ Cámara abierta con backend: {backend}")
            break

    if not cap or not cap.isOpened():
        raise Exception(f"❌ No se pudo acceder a la cámara en el índice {video_source}.")

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    reconocido = False

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("❌ No se pudo capturar el frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            rostro = gray[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            label, confidence = face_recognizer.predict(rostro)

            if confidence < threshold:
                cv2.putText(frame, f'{imagePaths[label]} ({confidence:.2f})', (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                reconocido = True
            else:
                cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('🧐 Reconociendo Rostros', frame)

        if cv2.waitKey(1) == 27:  # Salir con ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    return reconocido


if __name__ == "__main__":
    dataPath = r'C:\Users\DELL\OneDrive\Desktop\entrenar_ia\data'
    personName = 'Guardado'

    try:
        print("🔍 Intentando reconocer rostros...")
        reconocido = reconocer_rostros(dataPath)

        if not reconocido:
            print("❌ No se reconoció ningún rostro. Iniciando captura de nuevas imágenes...")
            capturar_rostros(personName, dataPath)

            print("🧠 Entrenando el modelo con las nuevas imágenes...")
            entrenar_modelo(dataPath)

            print("🔍 Intentando reconocer rostros nuevamente...")
            if reconocer_rostros(dataPath):
                print("✅ Rostro reconocido después del entrenamiento.")
            else:
                print("❌ Aún no se reconoce el rostro.")
        else:
            print("✅ Rostro reconocido. No es necesario capturar nuevas imágenes.")

    except Exception as e:
        print(f"❌ Error: {e}")
