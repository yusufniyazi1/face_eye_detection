import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade/frontalface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/eye.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 2
text_color = (0, 0, 255)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        text_origin = (x, y - int(h * 0.1))


        cv2.putText(frame, "Face", text_origin, font, font_scale, text_color, font_thickness)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)


            text_origin = (ex + int(ew / 2), ey + int(eh / 2))


            cv2.putText(roi_color, "Eyes", text_origin, font, font_scale, text_color, font_thickness)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
