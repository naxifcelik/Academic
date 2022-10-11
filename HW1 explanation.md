# F-Z437e
İTÜ 2022 güze dönemi FİZ437e dersi ödev ve projeleri
NAZİF ÇELİK 090200712 

HW-1 


VERİ SETİ OLUŞTURMA  

bing_image_dowloader isimli kütüphane ile internetden çoklu fotoğraf indirme işlemini yaptım. Elma ve araba fotoğraflarını kullandım veri seti için. OpenCV kütüphanesindeki cvtColor modülü ile fotoğrafları grayscale yaptım. Ayrıca fotoğrafları arraylere scale etme ve arraylere sığdırma işlemi için de numpy kütüphanesini kullandım.

 

KNN SINIFLANDIRICI 
İlk olarak resimleri numpy array formatına sığdırdım. Komşuların K sayısının öklid mesafesini hesapladım. Bu hesaba göre en yakın komşular arasında her kategorideki veri noktalarının sayısını hesapladım. Yeni veri noktalarını komşu sayısının maksimum olduğu kategoriye atadım. Modeli bu şekilde tamamladım ve KNN simülasyonunu verilerimin %90 ı ile eğitip %10 ile test ederek gerçekleştirdim. K sayısının bir fonksiyonu olarak eğitim ve test eğrisini çizdim.  

 

 

 
