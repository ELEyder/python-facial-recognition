import cv2
import os
import imutils

# Cambia por el nombre de la persona a reconocer
personName = 'Adriana'
# Cambia a la ruta donde hayas almacenado Data
dataPath = 'C:/Users/adria/OneDrive/Documentos/Engitronic-DESKTOP-TLCG3SD/Curso de Python/Clases/Clase 18/Proyecto Final/data'
personPath = dataPath + '/' + personName

# Si la carpeta no existe, se crea
if not os.path.exists(personPath):
	print('Carpeta creada: ',personPath)
	os.makedirs(personPath)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('Video.mp4')    #Usar si tienes un video listo

# Inicia el detector de rostros de Haarcascades
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0

while True:
        # Lee cada fotograma del video
	ret, frame = cap.read()
	if ret == False: break
	# Los redimensiona
	frame =  imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = frame.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)

        # Obtiene las coordenadas y las dimensiones de los rostros detectados
	for (x,y,w,h) in faces:
                # Dibuja el rectángulo alrededor de la cara
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		rostro = auxFrame[y:y+h,x:x+w]
		# Almacenaremos todos los rostros de un mismo tamaño
		rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),rostro)
		count = count + 1
	cv2.imshow('frame',frame)

	k =  cv2.waitKey(1)
	# Guarda 300 rostros de forma automática
	if k == 27 or count >= 300:
		break

cap.release()
cv2.destroyAllWindows()
