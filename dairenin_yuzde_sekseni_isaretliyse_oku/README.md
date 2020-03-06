Dairenin yüzde 80'i İşaretliyse Oku
===================================

Şablondaki dairelerin içerisinde sayılar mevcut. Bu sayılar kutucuğun belirli bir yüzdesini öntanımlı olarak doldurmakta.
İşaretleme yapılmayan kağıtları tespit etmek için, dairelerin doluluk oranlarına göre işaretlenip işaretlenmediğini tespit etmemiz gerekiyor.
Bu nedenle doluluk eşiğini %80 olarak belirlenip gerekli kontroller yapıldı.

```python
    # Bulunan konturların hangilerinin bubble olduğunu hesapla
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
```

Örnek:

`python3 main.py -i 20200305112230_004.png`
`python3 main.py -i 20200305112230_008.png`
