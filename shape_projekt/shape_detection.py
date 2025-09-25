# shape_detection.py
import cv2
import numpy as np
import os

# Bilder-Ordner (lege hier eigene Testbilder rein)
os.makedirs('results', exist_ok=True)

# Bild laden
image_path = 'test_image.jpg'  # Ersetze durch dein Bild
img = cv2.imread(image_path)
if img is None:
    print("Bild nicht gefunden:", image_path)
    exit()

# In Graustufen
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)


# Kanten erkennen
edges = cv2.Canny(gray, 100, 200)

# Konturen finden
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Konturen einzeichnen
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
        cv2.drawContours(img, [cnt], 0, (0,255,0), 2)  # Dreiecke grün
    elif len(approx) == 4:
        cv2.drawContours(img, [cnt], 0, (255,0,0), 2)  # Rechtecke blau
    else:
        cv2.drawContours(img, [cnt], 0, (0,0,255), 2)  # Andere Formen rot

# Ergebnisse anzeigen
cv2.imshow('Erkannt', img)
cv2.imshow('Kanten', edges)

# Ergebnis speichern
cv2.imwrite('results/result.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

