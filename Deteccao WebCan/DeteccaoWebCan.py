#Autor : Joao Henrique
import cv2

video = cv2.VideoCapture(0)

classificador_face = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

while True:
    conectado , frame = video.read()
    #print(conectado)

    frame_cinza = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador_face.detectMultiScale(frame_cinza, minSize = (50,50))

    # minSize tamanho minimo do quadrado, padrão é (30, 30)

    for(x, y, l, a) in faces_detectadas:
        cv2.rectangle(frame,(x , y), (x + l, y + a), (1, 1, 200), 2)
        #(x , y), (x + l, y + a), (1, 1, 255), 2)

    cv2.imshow("Video", frame)
    if (cv2.waitKey(1) == ord("q")):
        break

video.release()
cv2.destroyAllWindows()