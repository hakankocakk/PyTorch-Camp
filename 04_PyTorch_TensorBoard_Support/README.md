# PyTorch TensorBoard Support  

## Tensorboard Nedir?

TensorBoard, makine öğrenimi iş akışı sırasında karşılaştırmaların daha iyi yapılabilmesi adına kullanılan bir görselleştirme aracıdır. 
Tensorboard,
- Kayıp ve doğruluk gibi deney ölçümlerinin izlenmesine,
- Model grafiğinin görselleştirilmesine
- Yerleştirmelerin daha düşük boyutlu bir alana yansıtılmasına ve çok daha fazlasına olanak tanır.

Bu incelemede LeNet-5 modelinin bir varyantı eğitilecek olup Fashion-MNIST veri kümesi üzerinde çalışılacaktır.
Fashion-MNIST, farklı giysileri gösteren görüntü karolarından oluşan bir veri kümesidir. İçeriğindeki on sınıf etiketi, gösterilen giysi türünü belirtir.

```python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import numpy as np

# torch.utils.tensorboard sınıfı pytorch kullanırken tensorboard supporttan yararlanmamızı sağlayacaktır.
from torch.utils.tensorboard import SummaryWriter
```
## TensorBoard'da Görüntüleri Gösterme

Öncelikle veri setimizi modelde eğitmek için kullanmadan önce birkaç görselleştirme yaparak veriyi tanıyalım:

```python 
# Gather datasets and prepare them for consumption
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

```

Yukarıda görülen kodda kullanılan;

`transforms.Compose() :` Görüntüleri uygun bir formata dönüştürmek için kullanılır.

`transforms.ToTensor() :`Görüntüleri PyTorch tensörüne çevirir ve piksel değerlerini [0, 255] aralığından [0, 1] aralığına normalize eder. Bu işlem basitçe 

$X_{\text{new}} = \frac{X_{\text{old}}}{255}$ şeklinde yapılır.

`transforms.Normalize(mean, std) :`Sonuç olarak tüm piksel değerleri [-1, 1] aralığına getirilir, bu da sinir ağlarının daha stabil öğrenmesini sağlar.

$X_{\text{new}} = \frac{X_{\text{old}-μ}}{σ}$

`μ : ortalama    σ : standart sapma`



```python 
# Store separate training and validations splits in ./data
training_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
validation_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)


```
`torchvision.datasets.FashionMNIST(...): ` FashionNMIST veri setinin indirilmesini sağlar.
Yukarıdaki kod torchvision kütüphanesi kullanılarak FashionNMIST veri setinin öğrenme ve test veri seti olmak üzere iki sete ayrılmasını sağlayacaktır.

```python 
training_loader = torch.utils.data.DataLoader(training_set,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)
```

```python 
validation_loader = torch.utils.data.DataLoader(validation_set,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=2)```

```

`DataLoader(training_set, ...) :` Veri kümesini yükleyen nesne oluşturur.

`batch_size=4: ` Her iterasyonda 4 örnekten oluşan mini-batch'ler oluşturur. (Mini-batch eğitim verisinin modele küçük parçalar halinde verilmesi anlamına gelir ve eğitim hızını ile bellek kullanımını etkileyen önemli bir hiperparametredir.)

`shuffle=True/False :` Eğitim verilerininin karıştırılıp karıştırılmayacağını belirler. 

Eğitim setinde shuffleın true alınması verilerin karıştırılarak, modelin verileri ezberlemesini engellenmesi ve daha iyi genelleştirme yapılması amaçlanır.(Over-fitting engellenmiş olur.)

`num_workers=2 :` İki işlemci çekirdeği kullanarak verileri yükler. 

```python 
# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
```
Veri setindeki görseller yukarıdaki sınıflardan birine ait olacak şekilde tahminlenecektir.


```python 
# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
```
Yukarıda tanımlanan fonksiyon Pytorch tensorlerinin - ki bu durumda görsellerin verilerini tutuyorlar - matplotlib kütüphanesi yardımıyla görselleştirilmesini sağlar.

`img: ` Görselleştirilecek olan Pytorch tensorüdür.
`one_channel : `Görselin tek kanallı (gri tonlamalı) olup olmadığını belirten bir boolean değerdir. Eğer görsel tek kanallıysa gri renkli birden çok kanallı ise renkli bir görünüme sahip olacaktır.
---------------------------------------------------------------?
```python 
# Extract a batch of 4 images
dataiter = iter(training_loader)
images, labels = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
```
`iter(training_loader):` Eğitim veri seti üzerinde bir iterator oluşturur. Bu iterator, her çağrıldığında bir batch veri döndürülür.
`images, labels = next(dataiter):` Bir sonraki batch çağırılacaktır.

Sonrasında ise batchler bir gride gösterilecek şekilde aşağıdaki tabloda olduğu gibi bir çıktuya sahip olacaktır.

Örnek Çıktı:

![output](https://github.com/user-attachments/assets/8d2d9c44-c2f9-4956-8f1c-8595efb8303d)

Yukarıda, giriş verilerimizin bir mini-batch'ini görselleştirmek için TorchVision ve Matplotlib kullandık. İncelememizin devamında ise görüntüyü TensorBoard tarafından görüntülenmek üzere kaydetmek için `SummaryWriter` üzerindeki `add_image()` fonksiyonunu kullanılıp verinin hemen diske yazıldığından emin olmak için ise `flush()` çağrısı yapılacaktır.


```python 
writer = SummaryWriter('runs/fashion_mnist_experiment_1')


writer.add_image('Four Fashion-MNIST Images', img_grid)
writer.flush()

# To view, start TensorBoard on the command line with:
#   tensorboard --logdir=runs
# ...and open a browser tab to http://localhost:6006/
```


Eğer TensorBoard'u komut satırından başlatır ve yeni bir tarayıcı sekmesinde açarsanız (genellikle localhost:6006 adresinde), görsel ızgarasını IMAGES sekmesi altında görmelisiniz.


Bunun için komut satırına öncelikle: tensorboard --logdir=runs
yazılmalı daha sonrasında ise http://localhost:6006/ şeklinde bize geri dönütü verilen siteye tıklayarak görseli inceleyebiliriz.

Örnek Çıktı:
<img width="292" alt="clothimages" src="https://github.com/user-attachments/assets/485e946a-76bc-43e0-938e-9dc517740bbc" />


## Skalarları Grafikleştirerek Eğitimi Görselleştirme

TensorBoard, eğitimin ilerlemesini ve etkinliğini takip etmek için kullanışlıdır. Aşağıda, bir eğitim döngüsü çalıştıracağız, bazı metrikleri takip edeceğiz ve bu verileri TensorBoard'un kullanımı için kaydedeceğiz.




Görsel karelerimizi kategorilere ayırmak için bir model ve eğitim için bir optimizasyon algoritması ile loss fonksiyonunu tanımlayalım:

```python 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

Yukarıdaki kod ile LeNet-5 modelinin bir varyantını tanımlanır.

`nn.Conv2d():` evrişim katmanlarını tanımlar.

`nn.MaxPool2d():` maksimum havuzlama katmanlarını tanımlar.

`nn.Linear():` tam bağlantılı katmanları tanımlar.

`forward():` modelin ileri yayılımını (forward pass) tanımlar.

`nn.CrossEntropyLoss():` çapraz entropi kaybını hesaplar.

`optim.SGD():` stokastik gradyan iniş (SGD) optimizasyon algoritmasını tanımlar.


Şimdi, tek bir epoch boyunca eğitim yapalım ve her 1000 batch'te bir eğitim seti ile validation seti kayıplarını değerlendirelim:


```python 
print(len(validation_loader))
for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, data in enumerate(training_loader, 0):
        # basic training loop
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # Every 1000 mini-batches...
            print('Batch {}'.format(i + 1))
            # Check against the validation set
            running_vloss = 0.0
            
            net.train(False) # Don't need to track gradents for validation
            for j, vdata in enumerate(validation_loader, 0):
                vinputs, vlabels = vdata
                voutputs = net(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
            net.train(True) # Turn gradients back on for training
            
            avg_loss = running_loss / 1000
            avg_vloss = running_vloss / len(validation_loader)
            
            # Log the running loss averaged per batch
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch * len(training_loader) + i)

            running_loss = 0.0
print('Finished Training')

writer.flush()
```

<img width="526" alt="trainvsvalid" src="https://github.com/user-attachments/assets/7df7b490-42c2-4d3a-9d04-0d4e02178473" />





## Modeli Görselleştirme

TensorBoard, modeliniz içindeki veri akışını incelemek için de kullanılabilir. Bunu yapmak için, bir model ve örnek bir girdi ile add_graph() yöntemini çağırın. TensorBoard'u açtığınızda :

```python 
# Again, grab a single mini-batch of images
dataiter = iter(training_loader)
images, labels = next(dataiter)

# add_graph() will trace the sample input through your model,
# and render it as a graph.
writer.add_graph(net, images)
writer.flush()
```
TensorBoard'a geçtiğinizde, bir GRAPHS sekmesi görmelisiniz. Modeliniz içindeki katmanları ve veri akışını görmek için "NET" düğümüne çift tıklayın.


<img width="318" alt="graphs" src="https://github.com/user-attachments/assets/8bd67aa9-8e0c-4fd6-a60f-6204bb682d74" />




Aşağıda, verilerimizden bir örnek alacağız ve böyle bir gömme (embedding) oluşturacağız:
```python 
# Select a random subset of data and corresponding labels
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# Extract a random subset of data
images, labels = select_n_random(training_set.data, training_set.targets)

# get the class labels for each image
class_labels = [classes[label] for label in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.flush()
writer.close()
```
`add_embedding(): ` Kullandığımız 28x28 boyutundaki görsel kareler, 784 boyutlu vektörler olarak modellenebilir (28 * 28 = 784). Bu vektörleri daha düşük boyutlu bir temsile yansıtmak öğretici olabilir. add_embedding() yöntemi, bir veri kümesini en yüksek varyansa sahip üç boyuta yansıtır ve bunları etkileşimli bir 3B grafik olarak gösterir. add_embedding() yöntemi, bunu otomatik olarak en yüksek varyansa sahip üç boyuta yansıtarak gerçekleştirir.


Şimdi, TensorBoard'a geçip PROJECTOR sekmesini seçerseniz, projeksiyonun 3B temsilini görmelisiniz. Modeli döndürebilir ve yakınlaştırabilirsiniz.
Örnek Görüntü:

![resim](https://github.com/user-attachments/assets/ededef92-cb36-4e95-9fe2-7a9afe359767)





Not: Daha iyi görünürlük için şunları yapmanız önerilir:

  - Soldaki "Color by" (Renklendirme) açılır menüsünden "label" (etiket) seçeneğini seçin.

  - Üst kısımdaki Gece Modu (Night Mode) simgesine tıklayarak açık renkli görselleri koyu bir arka plan üzerinde görüntüleyin.

# Credits:
https://www.tensorflow.org/tensorboard/get_started?hl=tr
https://www.youtube.com/watch?v=VJW9wU-1n18&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=16
https://www.youtube.com/watch?v=6CEld3hZgqc&list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN&index=5
