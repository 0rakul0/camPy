import cv2

rastreador = cv2.TrackerCSRT_create()

#puxa a camera
cap = cv2.VideoCapture(0)
ok, frame = cap.read()

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#seleciona o frame que vai rastrear
bbox = cv2.selectROI(frame)


ok = rastreador.init(frame, bbox)

while True:
    ok, frame = cap.read()

    if not ok:
        break

    #joga para o rastreador as coordenadas inicias
    ok, bbox = rastreador.update(frame)
    
    if ok:
        #joga nos valores das coordenadas
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2, 1)
    else:
        cv2.putText(frame, 'Error no rastreamento', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

    #deixa a imagem em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rastreamento de faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('Rastreando, frame', frame)

    #parar quando precionar esc
    if cv2.waitKey(1) & 0xFF == 27:
        break   
    