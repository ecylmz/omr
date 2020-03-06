from imutils.perspective import four_point_transform
from imutils import contours
from pyzbar import pyzbar
import numpy as np
import argparse
import imutils
import cv2
import math

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

# işaretlenen dairelerin hangi rakam olduğunu bulmak için kullanılıyor
score_keys = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}

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

# Bulunan toplam skor alanına göre kırpma işlemi yap
x, y, w, h = cv2.boundingRect(total_score_contour)

total_score_contour_image = image[y:y+h, x:x+w]
total_score_contour_gray_image = gray[y:y+h, x:x+w]

# Kırpılan alana threshold uygula.
thresh = cv2.threshold(total_score_contour_gray_image, 0,
                       255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Kırpılan alandaki konturları bul
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Bulunan konturların hangilerinin bubble olduğunu hesapla
bubbles = []
for c in cnts:
    (_, _, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 29 and h >= 29 and ar >= 0.9 and ar <= 1.1:
        bubbles.append(c)

if len(bubbles) != 20:
    print("Hatalı İşaretleme!!!")
    print("Bulunan bubble sayısı:", len(bubbles))
    # cv2.drawContours(total_score_contour_image, bubbles, -1, (0, 0, 255), 3)
    # cv2.imshow("bulunan bubbles", total_score_contour_image)
    # cv2.waitKey(0)

# bulunan daireleri yukarıdan aşağıya doğru sırala
bubbles = contours.sort_contours(bubbles, method="top-to-bottom")[0]

score = ""

# Her satır 10 daire içerecek şekilde tüm satırları gez.
for (_, i) in enumerate(np.arange(0, len(bubbles), 10)):
    # her satırdaki ilk 10 daireyi soldan sağa sırala
    cnts = contours.sort_contours(bubbles[i:i + 10])[0]
    bubbled = None

    for (j, c) in enumerate(cnts):
        # yalnızca mevcut dairenin gözükmesi için bir maske oluştur
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        # maskeyi threshold resme uygula, ardından sıfır olmayan piksellerin sayısını say
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        (x, y, w, h) = cv2.boundingRect(c)
        area = math.pi * ((min(w, h) / 2) ** 2)
        percent_marked = total / area
        # eğer sıfır olmayan piksellerin sayısı bir önceki daireninkinden fazlaysa bu daire işaretlenmiştir kabul edilir.
        if (bubbled is None or total > bubbled[0]) and percent_marked >= 0.8:  # pylint: disable=E1136
            bubbled = (total, j)

    if bubbled:
        score += str(score_keys[bubbled[1]])
        # draw the outline of the bubbled
        color = (0, 0, 255)
        cv2.drawContours(total_score_contour_image, [cnts[bubbled[1]]], -1, color, 3)

print("Score:", score)

cv2.imshow("bubbleds", total_score_contour_image)
cv2.waitKey(0)