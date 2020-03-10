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

# maskedeki konturları bulmak için bir takım ön işlemler
exam_status_mask_image = cv2.imread("exam_status_mask.png")
exam_status_mask_gray_image = cv2.cvtColor(
    exam_status_mask_image, cv2.COLOR_BGR2GRAY)
exam_status_mask_blurred_image = cv2.GaussianBlur(
    exam_status_mask_gray_image, (5, 5), 0)
exam_status_mask_edged_image = cv2.Canny(
    exam_status_mask_blurred_image, 75, 200)

# Sınava girmedi kutucuğunun yeri
exam_status_x, exam_status_y = 1485, 245
exam_status_h = 45
exam_status_w = 45
exam_status_box = image[exam_status_y:exam_status_y +
                        exam_status_h, exam_status_x:exam_status_x + exam_status_w]

# kağıt taranırken kağıttan kağıda ufak koordinat farklılıkları için hata payı bırak
exam_status_x_error, exam_status_y_error = 10, 10
exam_status_h_error, exam_status_w_error = 5, 5

# kağıttaki tüm konturları bul
cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

exam_status_contour = None

if len(cnts) > 0:
    # Bulunan konturları alanlarına göre küçükten büyüğe sırala.
    cnts = sorted(cnts, key=cv2.contourArea)
    for c in cnts:
        # konturun alanı 600'dan büyükse işlem yap
        if cv2.contourArea(c) > 300:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # dikdörtgen bir alan arıyoruz
            if len(approx) == 4:
                # 4 kenarı olan her konturu aradığımız bölge mi diye kontrol ediyoruz
                (x, y, w, h) = cv2.boundingRect(approx)
                if ((exam_status_x - exam_status_x_error <= x <= exam_status_x + exam_status_x_error) and
                        (exam_status_y - exam_status_y_error <= y <= exam_status_y + exam_status_y_error)):
                    img = image[y:y + h, x:x + w]
                    # Şartları sağlayan ve en küçük alana sahip bölge bizim işlem
                    # yapacağımız bölge olacak. Diğerlerine ihtiyaç yok.
                    exam_status_contour = approx
                    break

# Bulunan toplam skor alanına göre kırpma işlemi yap
x, y, w, h = cv2.boundingRect(exam_status_contour)

exam_status_contour_image = image[y:y+h, x:x+w]
exam_status_contour_gray_image = gray[y:y+h, x:x+w]

# Kırpılan alana threshold uygula.
thresh = cv2.threshold(exam_status_contour_gray_image, 0,
                       255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# MASKE İŞLEMLERİ
# Maske konturlarını bul
mask_thresh = cv2.threshold(exam_status_mask_gray_image, 0,
                            255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

mask_cnts = cv2.findContours(
    mask_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_cnts = imutils.grab_contours(mask_cnts)

mask_cnts = sorted(mask_cnts, key=cv2.contourArea, reverse=True)

# Bulunan konturların hangilerinin bubble olduğunu hesapla
mask_bubbles = []
for c in mask_cnts:
    (_, _, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 29 and h >= 29 and ar >= 0.9 and ar <= 1.1:
        mask_bubbles.append(c)

if len(mask_bubbles) != 1:
    print("Hatalı Maske!!!")
    print("Bulunan bubble sayısı:", len(mask_bubbles))

# MASKE İŞLEMLERİ SONU

cnts = contours.sort_contours(mask_bubbles)[0]
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
    # draw the outline of the bubbled
    cv2.drawContours(exam_status_contour_image, [
        cnts[bubbled[1]]], -1, (0, 0, 255), 3)

    print("Sınava Girmedi!")
else:
    print("Sınava Girdi.")

cv2.imshow("bubbleds", exam_status_contour_image)
cv2.waitKey(0)
