import cv2
import os
import imutils

def capturar_rostros(personName, dataPath, max_count=300, video_source=1):
    """
    Captura rostros de un video y los guarda en la ruta especificada.

    Parámetros:
    - personName: Nombre de la persona (se usará para crear una carpeta con su nombre).
    - dataPath: Ruta base donde se almacenarán las imágenes.
    - max_count: Número máximo de imágenes a capturar (por defecto 300).
    - video_source: Fuente de video (0 para cámara predeterminada, 1 para cámara externa, o ruta de un archivo de video).
    """
    # Crear la ruta completa para la carpeta de la persona
    personPath = os.path.join(dataPath, personName)

    # Verificar si la carpeta existe; si no, crearla
    if not os.path.exists(personPath):
        print('Carpeta creada:', personPath)
        os.makedirs(personPath)

    # Inicializar la captura de video
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)

    # Cargar el clasificador de caras
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0  # Contador de imágenes capturadas

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Si no hay más frames, salir del bucle

        # Redimensionar el frame y convertirlo a escala de grises
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        # Detectar caras en el frame
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Dibujar un rectángulo alrededor del rostro detectado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recortar el rostro y guardarlo como imagen
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
            count += 1

        # Mostrar el frame en una ventana
        cv2.imshow('frame', frame)

        # Salir si se presiona la tecla ESC o se alcanza el número máximo de imágenes
        k = cv2.waitKey(1)
        if k == 27 or count >= max_count:
            break

    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

    print(f"Proceso completado. Se capturaron {count} imágenes.")

