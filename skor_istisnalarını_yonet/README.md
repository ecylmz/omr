Skor İstisnalarını Yönet
========================

Skoru herhangi bir kontrol yapmadan `score` değişkenine yazıyoruz. Eksik olan satırlar için `?` karakteri koyuyoruz.
Daha sonra `score` değişkeninin `?` karakteri içerip içermediğine, 2 basamaklı olup olmadığına eğer 3 basamaklı ise 100 olup olmadığına bakılıyoruz.

Yöntem aşağıdaki gibi:

```python
score = ""

bubbled = [[], []]

# Her satır 10 daire içerecek şekilde tüm satırları gez.
for (k, i) in enumerate(np.arange(0, len(mask_bubbles), 10)):
    # her satırdaki ilk 10 daireyi soldan sağa sırala
    cnts = contours.sort_contours(mask_bubbles[i:i + 10])[0]

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

        if percent_marked >= 0.8:
            bubbled[k].append((total, j))

    if bubbled[k]:
        for b in bubbled[k]:
            score += str(score_keys[b[1]])

            # draw the outline of the bubbled
            color = (0, 0, 255)
            cv2.drawContours(total_score_contour_image, [cnts[b[1]]], -1, color, 3)
    else:
        score += "?"

print("Score:", score)

if ("?" in score) or len(score) > 3:
    print("Hatalı skor!")
else:
    if (len(score) == 2) or ((len(score) == 3) and score == "100"):
        print("Doğru skor!")
    else:
        print("Hatalı skor!")
```

Örnek:

`python3 main.py -i 20200305112230_007.png`
