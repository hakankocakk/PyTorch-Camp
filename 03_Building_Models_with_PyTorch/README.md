{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Building Models in PyTorch\n",
        "\n",
        "Bu bölümde PyTorch'ta derin öğrenme modelleri kurarken faydalanabileceğimiz bazı araçları inceleyeceğiz.\n",
        "\n",
        "## `torch.nn.Module` ve `torch.nn.Parameter`\n",
        "\n",
        "PyTorch’ta neredeyse tüm model öğeleri, `torch.nn.Module` sınıfından türetilmiştir. Bu sınıf, PyTorch modellerine özgü davranışları kapsüller.\n",
        "\n",
        "Önemli bir özellik: Eğer bir `Module` alt sınıfında öğrenme parametreleri varsa (`weights` gibi), bunlar `torch.nn.Parameter` tipinde ifade edilir. `Parameter`, `Tensor`’un bir alt sınıfıdır, fakat bir `Module` içinde attribute olarak tanımlanınca otomatik olarak modüle kaydedilir. `Module`’ün `parameters()` metodundan da bu parametrelere erişebilirsiniz.\n",
        "\n",
        "Aşağıda basit bir model tanımlayıp (`TinyModel`), parametrelerini listeleyeceğiz. Modelimizde iki doğrusal (linear) katman ve bir aktivasyon fonksiyonu bulunacak:\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import torch\n",
        "\n",
        "class TinyModel(torch.nn.Module):\n",
        "    \n",
        "    def __init__(self):\n",
        "        super(TinyModel, self).__init__()\n",
        "        \n",
        "        self.linear1 = torch.nn.Linear(100, 200)\n",
        "        self.activation = torch.nn.ReLU()\n",
        "        self.linear2 = torch.nn.Linear(200, 10)\n",
        "        self.softmax = torch.nn.Softmax(dim=1)  # Softmax, genelde son boyut üzerinde alınır\n",
        "    \n",
        "    def forward(self, x):\n",
        "        x = self.linear1(x)\n",
        "        x = self.activation(x)\n",
        "        x = self.linear2(x)\n",
        "        x = self.softmax(x)\n",
        "        return x\n",
        "\n",
        "tinymodel = TinyModel()\n",
        "\n",
        "print('The model:')\n",
        "print(tinymodel)\n",
        "\n",
        "print('\\n\\nJust one layer:')\n",
        "print(tinymodel.linear2)\n",
        "\n",
        "print('\\n\\nModel params:')\n",
        "for param in tinymodel.parameters():\n",
        "    print(param)\n",
        "\n",
        "print('\\n\\nLayer params:')\n",
        "for param in tinymodel.linear2.parameters():\n",
        "    print(param)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Yukarıdaki örnekte şunlara dikkat edin:\n",
        "* `__init__()` içerisinde katmanlar ve bileşenler tanımlanır.\n",
        "* `forward()` metodunda asıl hesaplama yapılır.\n",
        "* Modeli veya alt modüllerini yazdırdığınızda katman yapısını görebilirsiniz.\n",
        "* Parametreler `tinymodel.parameters()` veya `tinymodel.linear2.parameters()` üzerinden listelenir.\n",
        "\n",
        "## Yaygın Katman Tipleri\n",
        "\n",
        "### Doğrusal (Linear) Katmanlar\n",
        "\n",
        "En temel sinir ağı katmanı, *doğrusal (fully connected)* katmandır. Bu katman, her giriş öğesiyle her çıkış öğesini bir matris çarpımı yoluyla ilişkilendirir. Eğer modelin *m* girişi ve *n* çıkışı varsa, ağırlıklar *m x n* boyutlu bir matrisi temsil eder. Aşağıdaki örnekte 3 boyutlu girdi, 2 boyutlu çıktı veren bir `Linear` katman görüyoruz:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "lin = torch.nn.Linear(3, 2)\n",
        "x = torch.rand(1, 3)\n",
        "print('Input:')\n",
        "print(x)\n",
        "\n",
        "print('\\n\\nWeight and Bias parameters:')\n",
        "for param in lin.parameters():\n",
        "    print(param)\n",
        "\n",
        "y = lin(x)\n",
        "print('\\n\\nOutput:')\n",
        "print(y)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "* `lin.weight` ve `lin.bias` birer `Parameter`’dır ve autograd ile gradyan izlenir.\n",
        "* Doğrusal katmanlar sınıflandırma problemlerinde sık sık kullanılır (genellikle modelin sonunda, sınıf sayısı kadar nöron çıkar).\n",
        "\n",
        "### Konvolüsyonel (Convolutional) Katmanlar\n",
        "\n",
        "*Konvolüsyon katmanları*, uzamsal korelasyonu yüksek veriler için tasarlanmıştır. En yaygın kullanım alanı görüntü işleme (bilgisayarlı görü) olsa da, NLP gibi sıralı verilerde de kullanılabilir.\n",
        "\n",
        "Aşağıda `LeNet` gibi klasik bir konvolüsyonlu model örneği yer alıyor (kısaltılmış şekilde). Bu model, 32×32 boyutunda siyah-beyaz görüntülerde 1 kanal girdi ile çalışıyor.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import torch.nn.functional as F\n",
        "\n",
        "class LeNet(torch.nn.Module):\n",
        "\n",
        "    def __init__(self):\n",
        "        super(LeNet, self).__init__()\n",
        "        # 1 giriş kanalı (siyah-beyaz), 6 çıkış kanalı, 5x5 konvolüsyon çekirdeği (kernel)\n",
        "        self.conv1 = torch.nn.Conv2d(1, 6, 5)\n",
        "        self.conv2 = torch.nn.Conv2d(6, 16, 3)\n",
        "        # Lineer katmanlar\n",
        "        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  \n",
        "        self.fc2 = torch.nn.Linear(120, 84)\n",
        "        self.fc3 = torch.nn.Linear(84, 10)\n",
        "\n",
        "    def forward(self, x):\n",
        "        # İlk konvolüsyondan sonra ReLU ve 2x2 max pool\n",
        "        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))\n",
        "        # İkinci konvolüsyondan sonra ReLU ve 2x2 max pool\n",
        "        x = F.max_pool2d(F.relu(self.conv2(x)), 2)\n",
        "        # 16x6x6 -> düzleştirme (flatten)\n",
        "        x = x.view(-1, self.num_flat_features(x))\n",
        "        x = F.relu(self.fc1(x))\n",
        "        x = F.relu(self.fc2(x))\n",
        "        x = self.fc3(x)\n",
        "        return x\n",
        "\n",
        "    def num_flat_features(self, x):\n",
        "        size = x.size()[1:]  # batch boyutu hariç diğer boyutlar\n",
        "        num_features = 1\n",
        "        for s in size:\n",
        "            num_features *= s\n",
        "        return num_features"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "**Konvolüsyon katmanı** argümanları:\n",
        "* Birinci argüman = giriş kanal sayısı (ör. siyah-beyaz ise `1`, renkli RGB ise `3`)\n",
        "* İkinci argüman = konvolüsyondan sonra üretilecek *feature map* (özellik haritası) sayısı.\n",
        "* Üçüncü argüman = konvolüsyon çekirdeği (kernel) boyutu. Tek sayı girerseniz kare olarak (örn. `5` → 5×5), tuple olarak da farklı en-boy verebilirsiniz.\n",
        "\n",
        "**Max Pooling** işlemi, konvolüsyon katmanından çıkan aktivasyon haritasını küçülterek hesaplama yükünü azaltır ve önemli özelliği korur. `(2,2)` boyutlu havuzlama ile 2×2 blokların en büyük değeri alınır.\n",
        "\n",
        "PyTorch; 1D, 2D ve 3D veriler için farklı konvolüsyon katmanı çeşitlerini ve isteğe göre **stride**, **padding** gibi parametreleri destekler.\n",
        "\n",
        "### RNN (Recurrent) Katmanlar\n",
        "\n",
        "*Yinelemeli sinir ağları*, sıralı verilerle (zaman serileri, metin vb.) çalışır. LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit) gibi varyantları da vardır. Aşağıda bir LSTM tabanlı kelime etiketleme (POS Tagging) örneği görebilirsiniz:\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "class LSTMTagger(torch.nn.Module):\n",
        "\n",
        "    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):\n",
        "        super(LSTMTagger, self).__init__()\n",
        "        self.hidden_dim = hidden_dim\n",
        "\n",
        "        # Kelime gömme (embedding) katmanı\n",
        "        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)\n",
        "\n",
        "        # LSTM katmanı\n",
        "        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)\n",
        "\n",
        "        # Gizli durumdan (hidden state) etiket alanına lineer dönüşüm\n",
        "        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)\n",
        "\n",
        "    def forward(self, sentence):\n",
        "        embeds = self.word_embeddings(sentence)\n",
        "        # LSTM'e veriyoruz, çıktısı (lstm_out, (h, c))\n",
        "        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))\n",
        "        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))\n",
        "        tag_scores = F.log_softmax(tag_space, dim=1)\n",
        "        return tag_scores"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Yukarıda:\n",
        "* `vocab_size`: kelime dağarcığı büyüklüğü (indislerle temsil)\n",
        "* `embedding_dim`: kelimeleri gömmek için kullanılan boyut sayısı\n",
        "* `hidden_dim`: LSTM'in gizli durumu (memory) boyutu\n",
        "* `tagset_size`: kaç farklı etiket (POS etiketi) olduğu\n",
        "\n",
        "Girdi (bir cümle) kelime indislerinden oluşur. `Embedding` katmanı, her kelimeyi `embedding_dim` boyutlu bir vektöre dönüştürür. LSTM bu vektör dizisini işleyip her kelime için `hidden_dim` boyutunda bir çıktı üretir. Son olarak lineer katman ve `log_softmax` ile sınıflandırma yapılır.\n",
        "\n",
        "Daha fazlası için [Sequence Models and LSTM Networks](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) rehberine bakabilirsiniz.\n",
        "\n",
        "### Transformer Katmanları\n",
        "\n",
        "*Transformers*, özellikle doğal dil işleme (NLP) alanında son yıllarda en iyi sonuçları veren evrişimlere benzer. BERT gibi modeller, Transformer mimarisinin türevleridir. PyTorch, `Transformer` sınıfını sunarak çok kafalı (multi-head) dikkat (attention) mekanizmaları, kodlayıcı/çözücü katmanları vb. parametrelerle kurmanıza izin verir.\n",
        "Ayrıca alt bileşenleri (`TransformerEncoder`, `TransformerDecoder`, `TransformerEncoderLayer`, `TransformerDecoderLayer`) de mevcuttur. Daha fazlası için [dökümantasyon](https://pytorch.org/docs/stable/nn.html#transformer) ve ilgili [tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) incelenebilir.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Diğer Katmanlar ve Fonksiyonlar\n",
        "\n",
        "### Veri Manipülasyon Katmanları\n",
        "\n",
        "**Max pooling** gibi işlemler öğrenme sürecinde parametre gerektirmezler; tensörü küçültüp belirli bir boyutta en büyük değeri seçerler. \n",
        "\n",
        "Örneğin:\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "my_tensor = torch.rand(1, 6, 6)\n",
        "print(my_tensor)\n",
        "\n",
        "maxpool_layer = torch.nn.MaxPool2d(3)\n",
        "print(maxpool_layer(my_tensor))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Burada `6x6` girdi, `3x3` bir havuzlama uyguladı ve çıktısı `2x2` boyutlu oldu. Her 3×3 bloktaki en büyük değeri seçer.\n",
        "\n",
        "**Normalization (BatchNorm vb.)** katmanları, bir katmanın çıktısını ortalayıp belirli bir standart sapma çevresinde ölçekler. Bu, modelin eğitimini stabilize eder ve öğrenmeyi hızlandırır.\n",
        "\n",
        "Örnek:\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "my_tensor = torch.rand(1, 4, 4) * 20 + 5\n",
        "print(my_tensor)\n",
        "\n",
        "print(my_tensor.mean())\n",
        "\n",
        "norm_layer = torch.nn.BatchNorm1d(4)\n",
        "normed_tensor = norm_layer(my_tensor)\n",
        "print(normed_tensor)\n",
        "\n",
        "print(normed_tensor.mean())"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Bu örnekte tensörün ortalaması yaklaşık 15 civarındayken, BatchNorm sonrası ortalama neredeyse 0’a yaklaşıyor. Aktivasyon fonksiyonlarının merkez civarındaki duyarlılıkları daha yüksek olduğu için bu durum öğrenmeye yardımcı olur.\n",
        "\n",
        "**Dropout** katmanı, giriş tensöründeki bazı değerleri eğitim sırasında rastgele 0'a ayarlar (belirli bir olasılıkla), böylece modelin belirli nöronlara aşırı bağımlılık geliştirmesi engellenir (regularization). Tahmin (inference) aşamasında dropout kapalıdır.\n",
        "\n",
        "Örnek:\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "my_tensor = torch.rand(1, 4, 4)\n",
        "\n",
        "dropout = torch.nn.Dropout(p=0.4)\n",
        "print(dropout(my_tensor))\n",
        "print(dropout(my_tensor))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Her çalıştırmada farklı elemanların 0’a gittiğini görebilirsiniz. `p` parametresi, herhangi bir elemanın 0’a dönüşme olasılığını belirtir.\n",
        "\n",
        "### Aktivasyon Fonksiyonları\n",
        "\n",
        "Aktivasyon fonksiyonları, ağı lineer olmayan hale getirerek derin öğrenmeyi mümkün kılar. Sadece matris çarpımı yapsaydık, çok katmanlı olsa dahi aslında tek bir lineer dönüşüme indirgenebilirdi.\n",
        "\n",
        "PyTorch, ReLU ailesi, Tanh, Sigmoid, Softmax vb. birçok aktivasyon fonksiyonunu `torch.nn` içinde sunar.\n",
        "\n",
        "### Kayıp Fonksiyonları (Loss Functions)\n",
        "\n",
        "Kayıp fonksiyonları, modelin tahmini ile gerçek değer arasındaki farkı ölçer. PyTorch’ta `torch.nn.MSELoss` (L2 norm), `CrossEntropyLoss`, `NLLLoss` (negatif olasılık lojitiği), vb. pek çok kayıp fonksiyonu vardır.\n",
        "\n",
        "## İleri Seviye: Katmanların Yerine Geçmek\n",
        "\n",
        "PyTorch, ağda bir katmanı veya fonksiyon bloğunu değiştirebilmeniz için esnek bir yapı sunar. Kendi özel katmanlarınızı `nn.Module`'u miras alarak tanımlayabilir, `forward()` içinde istediğiniz hesaplamayı yapabilirsiniz. Örneğin, custom bir katman oluşturmak veya PyTorch'un sağladığı bir katmanı belirli bir aşamada override etmek kolaydır.\n",
        "\n",
        "Bu şekilde, PyTorch’ta katmanları tanımlayıp birleştirerek istediğiniz mimariyi inşa edebilirsiniz. Kolay gelsin!"
      ]
    }
  ]
}
