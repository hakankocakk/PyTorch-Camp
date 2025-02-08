# Building Models in PyTorch

Bu bölümde PyTorch'ta derin öğrenme modelleri kurarken faydalanabileceğimiz bazı araçları inceleyeceğiz.

## torch.nn.Module ve torch.nn.Parameter

PyTorch’ta neredeyse tüm model öğeleri, `torch.nn.Module` sınıfından türetilmiştir. Bu sınıf, PyTorch modellerine özgü davranışları kapsüller.

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

## Yaygın Katman Tipleri

### Doğrusal (Linear) Katmanlar

En temel sinir ağı katmanı, *doğrusal (fully connected)* katmandır. Bu katman, her giriş öğesiyle her çıkış öğesi arasında bir matris çarpımı yapar.

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

### Konvolüyonel (Convolutional) Katmanlar

Konvolüyon katmanları, uzamsal korelasyonu yüksek veriler için tasarlanmıştır. Bilgisayarlı görü alanında yaygın olarak kullanılır.

```python
import torch.nn.functional as F

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

### RNN (Recurrent) Katmanlar

Yinelemeli sinir ağları (RNN'ler), sıralı verilerle çalışır. LSTM ve GRU gibi türevleri vardır.

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
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
```

## Diğer Katmanlar ve Fonksiyonlar

### Veri Manipülasyon Katmanları

**Max pooling** gibi işlemler tensörü küçültmek için kullanılır:

```python
my_tensor = torch.rand(1, 6, 6)
maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))
```

**Dropout** katmanı, rastgele nöronları 0'a ayarlar:

```python
my_tensor = torch.rand(1, 4, 4)
dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
```

## Aktivasyon Fonksiyonları ve Kayıp Fonksiyonları

PyTorch, ReLU, Tanh, Sigmoid, Softmax gibi aktivasyon fonksiyonlarını destekler. Kayıp fonksiyonları olarak `torch.nn.MSELoss`, `torch.nn.CrossEntropyLoss`, `torch.nn.NLLLoss` gibi seçenekler vardır.

## Özel Katmanlar Tanımlamak

PyTorch, `nn.Module`'u miras alarak kendi katmanlarınızı tanımlamanıza olanak tanır.

Bu şekilde PyTorch ile çeşitli sinir ağı modelleri kurabilir, optimize edebilir ve eğitebilirsiniz. Kolay gelsin!





