Klasik Soru ve Cevap Kağıdı
===========================

Yöntemler
---------

### Skor alanını bulmak

Kontur bulunduktan sonra hangi konturun hangi sırada olduğunu bildiğimizi varsayıyorum. Bu çok doğru bir yaklaşım değil.
Doğru yaklaşım şu olmalı:

Koordinati bilinen bir alan içerisinde skor alanı var mı kontrolü olmalı.

## Daire bulmak

Skor alanı içerisindeki daireleri bulmak için:

```python
# loop over the contours
for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
```

Bu yaklaşımın doğruluğu örneklerle araştırılmalı. Buradaki 20 değeri ise deneysel olarak bulundu. Taranan sayfanın
çözünürlüğüne göre değişkenlik gösterebilir.

Test için:

```bash
python3 main.py -i qr1.png
```
