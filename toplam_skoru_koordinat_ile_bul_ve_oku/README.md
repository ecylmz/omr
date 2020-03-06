Toplam Skor Alanını Koordinat ile Bul ve Oku
============================================

Toplam skor alanının koordinatları deneysel olarak bulunup, sabit olarak verildi.
Sayfadaki tüm konturlar bulunup konturun verilen koordinatlar içerisinde olup olmadığı kontrolü yapılmakta.

Belirlenen alan içerisindeki daireleri bulmak için aşağıdaki hesaplama yöntemi kullanıldı:

```python
# Bulunan konturların hangilerinin bubble olduğunu hesapla
bubbles = []
for c in cnts:
    (_, _, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 29 and h >= 29 and ar >= 0.9 and ar <= 1.1:
        bubbles.append(c)
```

Örnek:

`python3 main.py -i 20200305112230_004.png`
