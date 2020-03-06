Üst Üste Yerleştirme ile Skoru Oku
==================================

Skor alanını boş bir şablondan çıkardıktan sonra verilen resimdeki skor alanının üstüne yerleştiriyorum.
Şablondaki dairelerin konumlarının %80 dolu olup olmadığına bakıyorum.

Bu yöntem taşma problemlerini engelliyor. İşaretleme diğer daireye taşsa dahi en çok yüzdeye sahip olanı alma şansımız oluyor.

Belirlenen alan içerisindeki daireleri bulmak için aşağıdaki hesaplama yöntemi kullanıldı:

```python
# MASKE İŞLEMLERİ
# Maske konturlarını bul
mask_thresh = cv2.threshold(total_score_mask_gray_image, 0,
                       255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

mask_cnts = cv2.findContours(mask_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_cnts = imutils.grab_contours(mask_cnts)

mask_cnts = sorted(mask_cnts, key=cv2.contourArea, reverse=True)

# Bulunan konturların hangilerinin bubble olduğunu hesapla
mask_bubbles = []
for c in mask_cnts:
    (_, _, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 29 and h >= 29 and ar >= 0.9 and ar <= 1.1:
        mask_bubbles.append(c)

if len(mask_bubbles) != 20:
    print("Hatalı Maske!!!")
    print("Bulunan bubble sayısı:", len(mask_bubbles))
    # cv2.drawContours(total_score_contour_image, bubbles, -1, (0, 0, 255), 3)
    # cv2.imshow("bulunan bubbles", total_score_contour_image)
    # cv2.waitKey(0)

# bulunan daireleri yukarıdan aşağıya doğru sırala
mask_bubbles = contours.sort_contours(mask_bubbles, method="top-to-bottom")[0]
# MASKE İŞLEMLERİ SONU

score = ""

# Her satır 10 daire içerecek şekilde tüm satırları gez.
for (_, i) in enumerate(np.arange(0, len(mask_bubbles), 10)):
    # her satırdaki ilk 10 daireyi soldan sağa sırala
    cnts = contours.sort_contours(mask_bubbles[i:i + 10])[0]
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
```

Örnek:

`python3 main.py -i 20200305112230_006.png`
