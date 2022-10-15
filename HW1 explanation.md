# F-Z437e
İTÜ 2022 güze dönemi FİZ437e dersi ödev ve projeleri
NAZİF ÇELİK 090200712 

HW-1 


VERİ SETİ OLUŞTURMA  

bing_image_dowloader isimli kütüphane ile internetden çoklu fotoğraf indirme işlemini yaptım. Uçak ve araba fotoğraflarını kullandım veri seti için. OpenCV kütüphanesindeki cvtColor modülü ile fotoğrafları grayscale yaptım. Ayrıca fotoğrafları arraylere scale etme ve arraylere sığdırma işlemi için de numpy kütüphanesini kullandım. Çözünürlüğü 100x100 olarak belirledim.

 

KNN SINIFLANDIRICI 
İlk olarak resimleri numpy array formatına sığdırdım. Okldi fonksiyonunda komşuların K sayısının öklid mesafesini hesapladım. Uzaklık fonksiyonunda bu hesaba göre en yakın komşular arasında her kategorideki veri noktalarının sayısını hesapladım. Classification fonksiyonunda yeni veri noktalarını komşu sayısının maksimum olduğu kategoriye atadım yani kısacası etiketleme işlemini yaptım. Modeli bu şekilde tamamladım ve KNN simülasyonunu verilerimin %90 ı ile eğitip %10 ile test ederek gerçekleştirdim. resultt fonksiyonunda başarı oranını yüzde olarak ekrana bastırıyorum. 

 

 
kodumun çıktısı 
OUTPUT=![output](https://user-images.githubusercontent.com/48800008/196001317-900197d8-370f-49fc-ba14-43f86797f6ff.png)

