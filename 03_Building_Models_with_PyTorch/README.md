# Building Models in PyTorch

Bu bölümde PyTorch'ta derin öğrenme modelleri kurarken faydalanabileceğimiz bazı araçları inceleyeceğiz.

## `torch.nn.Module` ve `torch.nn.Parameter`

PyTorch’ta neredeyse tüm model öğeleri, `torch.nn.Module` sınıfından türetilmiştir. Bu sınıf, PyTorch modellerine özgü davranışları kapsüller.

Önemli bir özellik: Eğer bir `Module` alt sınıfında öğrenme parametreleri varsa (`weights` gibi), bunlar `torch.nn.Parameter` tipinde ifade edilir. `Parameter`, `Tensor`’un bir alt sınıfıdır, fakat bir `Module` içinde attribute olarak tanımlanınca otomatik olarak modüle kaydedilir. `Module`’ün `parameters()` metodundan da bu parametrelere erişebilirsiniz.

Aşağıda basit bir model tanımlayıp (`TinyModel`), parametrelerini listeleyeceğiz. Modelimizde iki doğrusal (linear) katman ve bir aktivasyon fonksiyonu bulunacak:

```python
import torch

class TinyModel(torch.nn.Module):
    
    def __init__(self):
        super(TinyModel, self).__init__()
        
        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax(dim=1)  # Softmax, genelde son boyut üzerinde alınır
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)
Yukarıdaki örnekte şunlara dikkat edin:

__init__() içerisinde katmanlar ve bileşenler tanımlanır.
forward() metodunda asıl hesaplama yapılır.
Modeli veya alt modüllerini yazdırdığınızda katman yapısını görebilirsiniz.
Parametreler tinymodel.parameters() veya tinymodel.linear2.parameters() üzerinden listelenir.
Yaygın Katman Tipleri
Doğrusal (Linear) Katmanlar
En temel sinir ağı katmanı, doğrusal (fully connected) katmandır. Bu katman, her giriş öğesiyle her çıkış öğesini bir matris çarpımı yoluyla ilişkilendirir. Eğer modelin m girişi ve n çıkışı varsa, ağırlıklar m x n boyutlu bir matrisi temsil eder. Aşağıdaki örnekte 3 boyutlu girdi, 2 boyutlu çıktı veren bir Linear katman görüyoruz:

python
Kopyala
Düzenle
lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)
lin.weight ve lin.bias birer Parameter’dır ve autograd ile gradyan izlenir.
Doğrusal katmanlar sınıflandırma problemlerinde sık sık kullanılır (genellikle modelin sonunda, sınıf sayısı kadar nöron çıkar).
Konvolüsyonel (Convolutional) Katmanlar
Konvolüsyon katmanları, uzamsal korelasyonu yüksek veriler için tasarlanmıştır. En yaygın kullanım alanı görüntü işleme (bilgisayarlı görü) olsa da, NLP gibi sıralı verilerde de kullanılabilir.

Aşağıda LeNet gibi klasik bir konvolüsyonlu model örneği yer alıyor (kısaltılmış şekilde). Bu model, 32×32 boyutunda siyah-beyaz görüntülerde 1 kanal girdi ile çalışıyor.

python
Kopyala
Düzenle
import torch.nn.functional as F

class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 giriş kanalı (siyah-beyaz), 6 çıkış kanalı, 5x5 konvolüsyon çekirdeği (kernel)
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # Lineer katmanlar
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # İlk konvolüsyondan sonra ReLU ve 2x2 max pool
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # İkinci konvolüsyondan sonra ReLU ve 2x2 max pool
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 16x6x6 -> düzleştirme (flatten)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # batch boyutu hariç diğer boyutlar
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
Konvolüsyon katmanı argümanları:

Birinci argüman = giriş kanal sayısı (ör. siyah-beyaz ise 1, renkli RGB ise 3)
İkinci argüman = konvolüsyondan sonra üretilecek feature map (özellik haritası) sayısı.
Üçüncü argüman = konvolüsyon çekirdeği (kernel) boyutu. Tek sayı girerseniz kare olarak (örn. 5 → 5×5), tuple olarak da farklı en-boy verebilirsiniz.
Max Pooling işlemi, konvolüsyon katmanından çıkan aktivasyon haritasını küçülterek hesaplama yükünü azaltır ve önemli özelliği korur. (2,2) boyutlu havuzlama ile 2×2 blokların en büyük değeri alınır.

PyTorch; 1D, 2D ve 3D veriler için farklı konvolüsyon katmanı çeşitlerini ve isteğe göre stride, padding gibi parametreleri destekler.

RNN (Recurrent) Katmanlar
Yinelemeli sinir ağları, sıralı verilerle (zaman serileri, metin vb.) çalışır. LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit) gibi varyantları da vardır. Aşağıda bir LSTM tabanlı kelime etiketleme (POS Tagging) örneği görebilirsiniz:

python
Kopyala
Düzenle
class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # Kelime gömme (embedding) katmanı
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # LSTM katmanı
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # Gizli durumdan (hidden state) etiket alanına lineer dönüşüm
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # LSTM'e veriyoruz, çıktısı (lstm_out, (h, c))
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
Yukarıda:

vocab_size: kelime dağarcığı büyüklüğü (indislerle temsil)
embedding_dim: kelimeleri gömmek için kullanılan boyut sayısı
hidden_dim: LSTM'in gizli durumu (memory) boyutu
tagset_size: kaç farklı etiket (POS etiketi) olduğu
Girdi (bir cümle) kelime indislerinden oluşur. Embedding katmanı, her kelimeyi embedding_dim boyutlu bir vektöre dönüştürür. LSTM bu vektör dizisini işleyip her kelime için hidden_dim boyutunda bir çıktı üretir. Son olarak lineer katman ve log_softmax ile sınıflandırma yapılır.

Daha fazlası için Sequence Models and LSTM Networks rehberine bakabilirsiniz.

Transformer Katmanları
Transformers, özellikle doğal dil işleme (NLP) alanında son yıllarda en iyi sonuçları veren evrişimlere benzer. BERT gibi modeller, Transformer mimarisinin türevleridir. PyTorch, Transformer sınıfını sunarak çok kafalı (multi-head) dikkat (attention) mekanizmaları, kodlayıcı/çözücü katmanları vb. parametrelerle kurmanıza izin verir. Ayrıca alt bileşenleri (TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer) de mevcuttur. Daha fazlası için dökümantasyon ve ilgili tutorial incelenebilir.

Diğer Katmanlar ve Fonksiyonlar
Veri Manipülasyon Katmanları
Max pooling gibi işlemler öğrenme sürecinde parametre gerektirmezler; tensörü küçültüp belirli bir boyutta en büyük değeri seçerler.

Örneğin:

python
Kopyala
Düzenle
my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))
Burada 6x6 girdi, 3x3 bir havuzlama uyguladı ve çıktısı 2x2 boyutlu oldu. Her 3×3 bloktaki en büyük değeri seçer.

Normalization (BatchNorm vb.) katmanları, bir katmanın çıktısını ortalayıp belirli bir standart sapma çevresinde ölçekler. Bu, modelin eğitimini stabilize eder ve öğrenmeyi hızlandırır.

Örnek:

python
Kopyala
Düzenle
my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())
Bu örnekte tensörün ortalaması yaklaşık 15 civarındayken, BatchNorm sonrası ortalama neredeyse 0’a yaklaşıyor. Aktivasyon fonksiyonlarının merkez civarındaki duyarlılıkları daha yüksek olduğu için bu durum öğrenmeye yardımcı olur.

Dropout katmanı, giriş tensöründeki bazı değerleri eğitim sırasında rastgele 0'a ayarlar (belirli bir olasılıkla), böylece modelin belirli nöronlara aşırı bağımlılık geliştirmesi engellenir (regularization). Tahmin (inference) aşamasında dropout kapalıdır.

Örnek:

python
Kopyala
Düzenle
my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))
Her çalıştırmada farklı elemanların 0’a gittiğini görebilirsiniz. p parametresi, herhangi bir elemanın 0’a dönüşme olasılığını belirtir.

Aktivasyon Fonksiyonları
Aktivasyon fonksiyonları, ağı lineer olmayan hale getirerek derin öğrenmeyi mümkün kılar. Sadece matris çarpımı yapsaydık, çok katmanlı olsa dahi aslında tek bir lineer dönüşüme indirgenebilirdi.

PyTorch, ReLU ailesi, Tanh, Sigmoid, Softmax vb. birçok aktivasyon fonksiyonunu torch.nn içinde sunar.

Kayıp Fonksiyonları (Loss Functions)
Kayıp fonksiyonları, modelin tahmini ile gerçek değer arasındaki farkı ölçer. PyTorch’ta torch.nn.MSELoss (L2 norm), CrossEntropyLoss, NLLLoss (negatif olasılık lojitiği), vb. pek çok kayıp fonksiyonu vardır.

İleri Seviye: Katmanların Yerine Geçmek
PyTorch, ağda bir katmanı veya fonksiyon bloğunu değiştirebilmeniz için esnek bir yapı sunar. Kendi özel katmanlarınızı nn.Module'u miras alarak tanımlayabilir, forward() içinde istediğiniz hesaplamayı yapabilirsiniz. Örneğin, custom bir katman oluşturmak veya PyTorch'un sağladığı bir katmanı belirli bir aşamada override etmek kolaydır.

Bu şekilde, PyTorch’ta katmanları tanımlayıp birleştirerek istediğiniz mimariyi inşa edebilirsiniz. Kolay gelsin!

markdown
Kopyala
Düzenle

**Not**: Bu dosyayı bir `.md` olarak kaydedip GitHub projenizin kök dizinine (ya da istediğiniz yerde) koyabilir, README işlevi görmesini sağlayabilirsiniz. Kod blokları GitHub üzerinde vurgulu (syntax highlighting) şekilde gözükecektir.

---

### Dikkat Edilmesi Gerekenler

1. **Etkileşimli özellikler:** Jupyter Notebook içindeki etkileşimli hücreler (örneğin widget’lar, canlı grafikler vb.) bu Markdown sürümde bulunmaz; burada statik metin ve kod blokları yer alır.
2. **Kaynak dosya**: `.ipynb` dosyanızı saklayarak, ileride tekrar düzenlemeniz gerekirse orijinal notebook üzerinde değişiklik yapabilirsiniz. Markdown’a dönüşüm, statik bir versiyondur.
3. **Resim ve medya**: Notebook’ta gömülü resimler veya media varsa, bunlar `.md` içerisinde el ile düzenlenmelidir (örn. `![Resim Açıklaması](./dosya_yolu.png)` gibi). Burada örnek Notebook’ta resim yoktu; eğer olsaydı, nbconvert veya manuel ekleme gerekebilirdi.

Bu şekilde, GitHub README’nizde Jupyter Notebook içeriğinizi açıklamalarla paylaşabilirsiniz. Kolay gelsin!







