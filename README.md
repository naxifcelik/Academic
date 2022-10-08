# F-Z437e
İTÜ 2022 güze dönemi FİZ437e dersi ödev ve projeleri
NAZİF ÇELİK 090200712 

HW-1 


VERİ SETİ OLUŞTURMA  

bing_image_dowloader isimli kütüphane ile internetden çoklu fotoğraf indirme işlemini yaptım. Elma ve araba fotoğraflarını kullandım veri seti için. OpenCV kütüphanesindeki cvtColor modülü ile fotoğrafları grayscale yaptım.  

 

KNN SINIFLANDIRICI 

 

İlk olarak resimleri numpy array formatına sığdırdım. Sift ile resim üzerindeki kilit noktaları buldum. İyi eşleşmeler için bir maske oluşturdum. En iyi KNN eşleşmelerini buldum. Her eşlenen nokta için olasılığı 1 arttırdım eğer olasılık bir nokta için 5 den fazla olursa nesnemi işaretledim. KNN simülasyonunu verilerimin %90 ı ile eğitip %10 ile test ederek gerçekleştirdim. K sayısının bir fonksiyonu olarak eğitim ve test eğrisini çizdim.  

 

 

 
