Toplam Skor Alanını Koordinat ile Bul ve Oku
============================================

Toplam skor alanının koordinatları deneysel olarak bulunup, sabit olarak verildi.
Sayfadaki tüm konturlar bulunup konturun verilen koordinatlar içerisinde olup olmadığı kontrolü yapılmakta.

Belirlenen alan içerisindeki öncelikle daireler bulunuyor ardından bu dairelere tek tek maske uygulanıp hangilerinin dolu olduğu hesaplanıyor:

```python
# Bulunan konturların hangilerinin bubble olduğunu hesapla
bubbles = []
for c in cnts:
    (_, _, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 29 and h >= 29 and ar >= 0.9 and ar <= 1.1:
        bubbles.append(c)

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
        # eğer sıfır olmayan piksellerin sayısı bir önceki daireninkinden fazlaysa bu daire işaretlenmiştir kabul edilir.
        if bubbled is None or total > bubbled[0]:  # pylint: disable=E1136
            bubbled = (total, j)

    score += str(score_keys[bubbled[1]])
    # draw the outline of the bubbled
    color = (0, 0, 255)
    cv2.drawContours(total_score_contour_image, [cnts[bubbled[1]]], -1, color, 3)

print("Score:", score)
```

Örnek:

`python3 main.py -i 20200305112230_004.png`
