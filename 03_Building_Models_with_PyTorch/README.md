# Building Models in PyTorch

Bu bölümde PyTorch'ta derin öğrenme modelleri kurarken faydalanabileceğimiz bazı araçları inceleyeceğiz. Burada yer alan katman türleri (Linear, Convolutional, RNN vb.) PyTorch ekosisteminde en yaygın kullanılan yapılardır ve farklı veri tipleri ya da farklı görevler için tasarlanmıştır.

## torch.nn.Module ve torch.nn.Parameter

PyTorch’ta neredeyse tüm model öğeleri, `torch.nn.Module` sınıfından türetilmiştir. Bu sınıf, PyTorch modellerine özgü davranışları kapsüller. Modelinizi bu sınıfı miras alarak tanımlarsınız ve bu sayede katmanlar, parametreler gibi bileşenleri organize edebilirsiniz.

Önemli bir özellik: Eğer bir `Module` alt sınıfında öğrenme parametreleri varsa (weights gibi), bunlar `torch.nn.Parameter` tipinde ifade edilir. `Parameter`, `Tensor`’un bir alt sınıfıdır, fakat bir `Module` içinde attribute olarak tanımlanınca otomatik olarak modüle kaydedilir. `Module`’ün `parameters()` metodundan da bu parametrelere erişebilirsiniz.

### Basit Bir Model Tanımlama

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
```

Bu modelde:
- `__init__()` içerisinde model bileşenlerini (katmanlar, fonksiyonlar) tanımlıyoruz.
- `forward()` metodunda girdi verisinin katmanlardan hangi sırayla geçeceğini belirtiyoruz.

## Yaygın Katman Tipleri

### Doğrusal (Linear) Katmanlar

En temel sinir ağı katmanı, *doğrusal (fully connected)* katmandır. Bu katman, her giriş öğesiyle her çıkış öğesini bir matris çarpımı üzerinden ilişkilendirir. Yani `y = xW + b` formülünü uygular. Eğer modelin `m` girişi ve `n` çıkışı varsa, ağırlıklar `m x n` boyutlu bir matrisi (W) ve `n` boyutlu bir bias vektörünü (b) temsil eder. Bu parametreler, modelin öğrenme sürecinde güncellenir.

**Kullanım Alanları:**
- Temel sınıflandırma ve regresyon modellerinde en sonda sınıf sayısı kadar nöron çıkışı almak için kullanılır.
- Özellikle tam bağlı katman (Fully Connected Layer) olarak geçer ve genelde CNN ya da RNN yapılarının en sonunda sınıflandırma katmanı olarak yer alır.

Örnek kod:

```python
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
```

Burada giriş boyutu 3, çıkış boyutu 2 olan bir doğrusal katman tanımladık. Parametreler `lin.weight` ve `lin.bias` olarak kaydedilir. `forward` çağrıldığında `x` ile bu ağırlıklar üzerinden matris çarpımı + bias işlemi yapılır.

### Konvolüsyonel (Convolutional) Katmanlar

*Konvolüsyon katmanları*, uzamsal korelasyonu yüksek veriler için tasarlanmıştır. Özellikle görüntü işleme alanında, iki boyutlu veriler (yükseklik-genişlik) üzerinde kayarak çalışan filtreler (kernel) kullanır.

- **Filtre (kernel) boyutu**: Girdi verisinin üzerinden kayan küçük bir pencere boyutu.
- **Giriş kanal sayısı (in_channels)**: Örneğin siyah-beyaz bir resim için 1, RGB resim için 3.
- **Çıkış kanal sayısı (out_channels)**: Öğrenilen filtrelerin sayısıdır. Her bir filtre farklı bir özelliği yakalayacak şekilde öğrenilir.
- **Stride**: Filtrenin resim üzerinde kaçar piksel ilerleyeceğini belirtir.
- **Padding**: Resmin kenarlarına sıfır (veya başka bir değer) eklenerek boyutu korumak veya istenen şekilde hizalamak için kullanılır.

Aşağıda LeNet gibi klasik bir konvolüsyonlu model örneği yer alıyor (kısaltılmış şekilde). Bu model, 32×32 boyutunda siyah-beyaz görüntülerde 1 kanal girdi ile çalışıyor.

```python
import torch.nn.functional as F

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)  # (giriş kanal=1, çıkış kanal=6, kernel=5x5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3) # (giriş kanal=6, çıkış kanal=16, kernel=3x3)
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
```

**Konvolüsyon katmanı** argümanları:
- `in_channels`: girişteki kanal sayısı
- `out_channels`: konvolüsyondan sonra üretilecek özellik haritası sayısı
- `kernel_size`: konvolüsyon çekirdeğinin (filtresinin) boyutu
- `stride`, `padding`, `dilation` gibi ek parametreler de isteğe bağlıdır.

**Max Pooling** işlemi, konvolüsyon katmanından çıkan aktivasyon haritasını küçülterek hesaplama yükünü azaltır ve önemli özellikleri korur. (2,2) boyutlu havuzlama ile 2×2 blokların en büyük değeri alınır.

PyTorch; 1D, 2D ve 3D veriler için farklı konvolüsyon katmanı çeşitlerini ve isteğe göre **stride**, **padding** gibi parametreleri destekler.

### RNN (Recurrent) Katmanlar

*Yinelemeli sinir ağları*, sıralı verilerle (zaman serileri, metin vb.) çalışır. LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit) gibi varyantları da vardır. Her bir RNN katmanı, zaman boyunca hesaplanacak gizli durum (hidden state) bilgisine sahiptir.

- **hidden_dim**: RNN içindeki gizli durumun boyutu (örneğin 128, 256 vb.).
- **num_layers**: Birden fazla RNN katmanını üst üste koymak için.
- **LSTM** ve **GRU** gibi türevler, uzun süreli bağımlılıkları daha iyi yakalayabilir.

Aşağıda bir LSTM tabanlı kelime etiketleme (POS Tagging) örneği görebilirsiniz:

```python
class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # LSTM'e veriyoruz, çıktısı (lstm_out, (h, c))
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
```

Bu örnekte:
- `vocab_size`: kelime dağarcığı büyüklüğü (indislerle temsil)
- `embedding_dim`: kelimeleri gömmek için kullanılan boyut sayısı
- `hidden_dim`: LSTM'in gizli durumu (memory) boyutu
- `tagset_size`: kaç farklı etiket (POS etiketi) olduğu

Girdi (bir cümle) kelime indislerinden oluşur. Embedding katmanı, her kelimeyi `embedding_dim` boyutlu bir vektöre dönüştürür. LSTM bu vektör dizisini işleyip her kelime için `hidden_dim` boyutunda bir çıktı üretir. Son olarak lineer katman ve `log_softmax` ile sınıflandırma yapılır.

## Diğer Katmanlar ve Fonksiyonlar

### Veri Manipülasyon Katmanları

**Max pooling** gibi işlemler öğrenme sürecinde parametre gerektirmezler; tensörü küçültüp belirli bir boyutta en büyük değeri seçerler.

Örneğin:

```python
my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))
```

Burada 6x6 girdi, 3x3 bir havuzlama uyguladı ve çıktısı 2x2 boyutlu oldu. Her 3×3 bloktaki en büyük değeri seçer.

**Normalization (BatchNorm vb.)** katmanları, bir katmanın çıktısını ortalayıp belirli bir standart sapma çevresinde ölçekler. Bu, modelin eğitimini stabilize eder ve öğrenmeyi hızlandırır.

Örnek:

```python
my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())
```

Bu örnekte tensörün ortalaması yaklaşık 15 civarındayken, BatchNorm sonrası ortalama neredeyse 0’a yaklaşıyor. Aktivasyon fonksiyonlarının merkez civarındaki duyarlılıkları daha yüksek olduğu için bu durum öğrenmeye yardımcı olur.

**Dropout** katmanı, giriş tensöründeki bazı değerleri eğitim sırasında rastgele 0'a ayarlar (belirli bir olasılıkla), böylece modelin belirli nöronlara aşırı bağımlılık geliştirmesi engellenir (regularization). Tahmin (inference) aşamasında dropout kapalıdır.

Örnek:

```python
my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))
```

Her çalıştırmada farklı elemanların 0’a gittiğini görebilirsiniz. `p` parametresi, herhangi bir elemanın 0’a dönüşme olasılığını belirtir.

### Aktivasyon Fonksiyonları

Aktivasyon fonksiyonları, ağı lineer olmayan hale getirerek derin öğrenmeyi mümkün kılar. Sadece matris çarpımı yapsaydık, çok katmanlı olsa dahi aslında tek bir lineer dönüşüme indirgenebilirdi. ReLU, Tanh, Sigmoid, Softmax gibi pek çok seçenek mevcuttur.

### Kayıp Fonksiyonları (Loss Functions)

Kayıp fonksiyonları, modelin tahmini ile gerçek değer arasındaki farkı ölçer. PyTorch’ta `torch.nn.MSELoss` (L2 norm), `torch.nn.CrossEntropyLoss`, `torch.nn.NLLLoss` (negatif log olasılığı), vb. pek çok kayıp fonksiyonu vardır. Model eğitimi sırasında, geriye yayılım (backpropagation) için bu fonksiyonlar temel alınır.

## İleri Seviye: Katmanların Yerine Geçmek

PyTorch, ağda bir katmanı veya fonksiyon bloğunu değiştirebilmeniz için esnek bir yapı sunar. Kendi özel katmanlarınızı `nn.Module`'u miras alarak tanımlayabilir, `forward()` içinde istediğiniz hesaplamayı yapabilirsiniz. Örneğin, custom bir katman oluşturmak veya PyTorch'un sağladığı bir katmanı belirli bir aşamada override etmek kolaydır.

Böylece, PyTorch’ta katmanları tanımlayıp birleştirerek istediğiniz mimariyi inşa edebilirsiniz. İhtiyaca göre CNN, RNN, Transformer veya tamamen özel yapılar geliştirebilir, eğitim verinizi bu modellere besleyerek uçtan uca öğrenme gerçekleştirebilirsiniz. Kolay gelsin!





