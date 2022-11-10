import datetime, cv2

video = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_classifier = cv2.CascadeClassifier("haarcascade_smile.xml")

while True:
    _, img = video.read()
    copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMutliScale(gray, 1.3, 5)

    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)

    cv2.imshow("Test", img)

    if cv2.waitKey(10) == "q":
        break
