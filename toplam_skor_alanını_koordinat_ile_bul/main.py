from imutils.perspective import four_point_transform
from imutils import contours
from pyzbar import pyzbar
import numpy as np
import argparse
import imutils
import cv2

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# kontur bulmak için bir takım ön işlemler
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# Toplam skor alanının yeri
total_score_x, total_score_y = 1175, 120
total_score_h = 105
total_score_w = 360
total_score_box = image[total_score_y:total_score_y +
                        total_score_h, total_score_x:total_score_x + total_score_w]

cv2.imshow("total_score", total_score_box)
cv2.waitKey(0)

# kağıt taranırken kağıttan kağıda ufak koordinat farklılıkları için hata payı bırak
total_score_x_error, total_score_y_error = 10, 10
total_score_h_error, total_score_w_error = 5, 5

# kağıttaki tüm konturları bul
cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

total_score_contour = None

if len(cnts) > 0:
    # Bulunan konturları alanlarına göre küçükten büyüğe sırala.
    cnts = sorted(cnts, key=cv2.contourArea)
    for c in cnts:
        # konturun alanı 600'dan büyükse işlem yap
        if cv2.contourArea(c) > 600:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # dikdörtgen bir alan arıyoruz
            if len(approx) == 4:
                # 4 kenarı olan her konturu aradığımız bölge mi diye kontrol ediyoruz
                (x, y, w, h) = cv2.boundingRect(approx)
                if ((total_score_x - total_score_x_error <= x <= total_score_x + total_score_x_error) and
                    (total_score_y - total_score_y_error <= y <= total_score_y + total_score_y_error)):
                    img = image[y:y + h, x:x + w]
                    cv2.imshow("bulunan total_score", img)
                    cv2.waitKey(0)
                    # Şartları sağlayan ve en küçük alana sahip bölge bizim işlem
                    # yapacağımız bölge olacak. Diğerlerine ihtiyaç yok.
                    total_score_contour = approx
                    break
