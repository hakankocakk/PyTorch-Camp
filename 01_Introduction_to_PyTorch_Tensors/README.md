# Introduction to PyTorch Tensors  
# PyTorch Tensors

Tensörler, PyTorch'taki temel veri soyutlamasıdır.


```python
import torch
import math
```

## Creating Tensors

Bir tensör oluşturmanın en basit yolu,`torch.empty()` çağrısını kullanmaktır:


```python
x = torch.empty(3, 4)
print(type(x))
print(x)
```

    <class 'torch.Tensor'>
    tensor([[-3.7485e+08,  4.4488e-41, -1.3545e-11,  3.1730e-41],
            [-1.3518e-11,  3.1730e-41,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00, -1.3500e-11,  3.1730e-41]])



* `torch` modulü kullanılarak bir tensör oluşturuldu.
* Tensörümüz iki boyutlu olup, 3 satır ve 4 sütundan (3x4) oluşuyor.
* Döndürülen nesnenin türü`torch.Tensor`, yani `torch.FloatTensor`’ın bir takma adıdır; varsayılan olarak, PyTorch tensörleri 32-bit kayan noktalı sayılarla doldurulur. 
* Tensörü yazdırdığınızda rastgele görünen bazı değerlerle karşılaşılabilinir. `torch.empty()` bellekte yer ayırır, ancak herhangi bir başlangıç değeriyle doldurmaz—görülen değerler, tahsis sırasında bellekte ne olduğuna bağlıdır.



```python
zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)
```

    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    tensor([[0.3126, 0.3791, 0.3087],
            [0.0736, 0.4216, 0.0691]])


### Random Tensors and Seeding (Tohumlama)

`torch.manual_seed()` çağrısı, Modellerin öğrenme ağırlıkları gibi tensörleri rastgele değerlerle başlatmak yaygın bir uygulamadır. Ancak, özellikle araştırma ortamlarında, sonuçların tekrar üretilebilir olmasını sağlamak istenebilir.

Bu durumda, rastgele sayı üreticisinin seed (tohum) değerini manuel olarak ayarlamak gereklidir.


```python
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)
```

    tensor([[0.3126, 0.3791, 0.3087],
            [0.0736, 0.4216, 0.0691]])
    tensor([[0.2332, 0.4047, 0.2162],
            [0.9927, 0.4128, 0.5938]])
    tensor([[0.3126, 0.3791, 0.3087],
            [0.0736, 0.4216, 0.0691]])
    tensor([[0.2332, 0.4047, 0.2162],
            [0.9927, 0.4128, 0.5938]])


Yukarıdaki gözlemlenen şey şudur: random1 ve random3 aynı değerlere sahiptir, random2 ve random4 de aynı değerlere sahiptir.
Bunun nedeni, rastgele sayı üreteci (RNG) için seed (tohum) değerini manuel olarak ayarlamanın, onu sıfırlaması ve aynı rastgele işlemlerin aynı sonuçları üretmesini sağlamasıdır.

Bu, özellikle tekrar üretilebilir araştırmalar yapmak veya model eğitimlerini karşılaştırmak için önemlidir.

### Tensor Shapes
Genellikle iki veya daha fazla tensör üzerinde işlem yaparken, bunların aynı *shape* sahip olmaları gerekir.Bu durumda, PyTorch’un `torch.*_like()` yöntemlerini kullanabiliriz. Bu yöntemler, mevcut tensörleri baz alarak yeni tensörler oluşturmanın kolay bir yoludur.


```python
x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)
```

    torch.Size([2, 2, 3])
    tensor([[[-3.7485e+08,  4.4488e-41, -3.7485e+08],
             [ 4.4488e-41,  4.4842e-44,  0.0000e+00]],
    
            [[ 1.5695e-43,  0.0000e+00,  1.9011e-13],
             [ 3.1734e-41,  0.0000e+00,  1.4013e-45]]])
    torch.Size([2, 2, 3])
    tensor([[[ 1.8914e-13,  3.1734e-41,  0.0000e+00],
             [ 1.4013e-45,  0.0000e+00,  3.1730e-41]],
    
            [[ 1.5835e-43,  0.0000e+00, -1.3501e-11],
             [ 3.1730e-41, -1.3540e-11,  3.1730e-41]]])
    torch.Size([2, 2, 3])
    tensor([[[0., 0., 0.],
             [0., 0., 0.]],
    
            [[0., 0., 0.],
             [0., 0., 0.]]])
    torch.Size([2, 2, 3])
    tensor([[[1., 1., 1.],
             [1., 1., 1.]],
    
            [[1., 1., 1.],
             [1., 1., 1.]]])
    torch.Size([2, 2, 3])
    tensor([[[0.6128, 0.1519, 0.0453],
             [0.5035, 0.9978, 0.3884]],
    
            [[0.6929, 0.1703, 0.1384],
             [0.4759, 0.7481, 0.0361]]])


Çağırılan fonksiyonlar:
* `.empty_like()` → x ile aynı boyutlarda, ancak başlangıç değeri olmadan oluşturulmuş bir tensör
* `.zeros_like()` → x ile aynı boyutlarda, ancak tüm değerleri sıfır olan bir tensör
* `.ones_like()` → x ile aynı boyutlarda, ancak tüm değerleri bir olan bir tensör
* `.rand_like()` → x ile aynı boyutlarda, ancak rastgele değerler içeren bir tensör
Bunları .shape özelliği ile kontrol edebiliriz. Her çağrı, orijinal tensörle aynı şekle (shape) sahip bir tensör döndürür.


```python
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)
```

    tensor([[3.1416, 2.7183],
            [1.6180, 0.0073]])
    tensor([ 2,  3,  5,  7, 11, 13, 17, 19])
    tensor([[2, 4, 6],
            [3, 6, 9]])


Eğer veriniz zaten bir Python listesi veya tuple biçimindeyse, `torch.tensor()` kullanarak tensör oluşturmak en basit yöntemdir.

*Note: `torch.tensor()` fonksiyonu verinin bir kopyasını oluşturur.*

## Tensor Data Types
Bir tensörün temel veri türünü ayarlamanın en basit yolu, oluşturma sırasında isteğe bağlı bir argüman kullanmaktır.

* dtype=torch.int16 ayarlayarak a tensörünü oluşturduk:
* Veri tipini ayarlamanın diğer bir yolu .to() metodunu kullanmaktır: aşağıdaki kod bloğunda önce rastgele float değerlerden oluşan b tensörünü oluşturduk, ardından .to() metoduyla c tensörünü 32-bit tam sayı formatına dönüştürdük.


```python
a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)    # float64 → int32 dönüşümü
print(c)
```

    tensor([[1, 1, 1],
            [1, 1, 1]], dtype=torch.int16)
    tensor([[ 0.9956,  1.4148,  5.8364],
            [11.2406, 11.2083, 11.6692]], dtype=torch.float64)
    tensor([[ 0,  1,  5],
            [11, 11, 11]], dtype=torch.int32)


PyTorch’ta Kullanılabilir Veri Tipleri

* `torch.bool`
* `torch.int8`
* `torch.uint8`
* `torch.int16`
* `torch.int32`
* `torch.int64`
* `torch.half`
* `torch.float`
* `torch.double`
* `torch.bfloat`


## PyTorch Tensörleri ile Matematik ve Mantık



```python
ones = torch.zeros(2, 2) + 1  # 0 matrisi + 1 → Tüm elemanlar 1 olur
twos = torch.ones(2, 2) * 2   # 1 matrisi * 2 → Tüm elemanlar 2 olur
threes = (torch.ones(2, 2) * 7 - 1) / 2  # (7 matrisi - 1) / 2 → Tüm elemanlar 3 olur
fours = twos ** 2  # 2 ** 2 → Tüm elemanlar 4 olur
sqrt2s = twos ** 0.5  # 2 ** 0.5 → Tüm elemanlar √2 olur

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)
```

    tensor([[1., 1.],
            [1., 1.]])
    tensor([[2., 2.],
            [2., 2.]])
    tensor([[3., 3.],
            [3., 3.]])
    tensor([[4., 4.],
            [4., 4.]])
    tensor([[1.4142, 1.4142],
            [1.4142, 1.4142]])


* Tensörler, skalerlerle doğrudan işleme sokulabilir. Çıktılar tensör olarak döner, böylece birden fazla işlemi zincirleme yapılabilir.
* Tensörler aynı şekle sahip olduğu sürece doğrudan toplanabilir, çıkarılabilir, çarpılabilir veya bölünebilir.



```python
powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)
```

    tensor([[ 2.,  4.],
            [ 8., 16.]])
    tensor([[5., 5.],
            [5., 5.]])
    tensor([[12., 12.],
            [12., 12.]])



```python
# kod hücresini çalıştırdığınızda çalışma zamanı hatası (RuntimeError) alınır.
a = torch.rand(2, 3)
b = torch.rand(3, 2)

print(a * b)

#Bu hata, tensörlerin boyutlarının çarpma işlemi için uyuşmaması nedeniyle oluşur.
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-52-ecf8b3a3ff0a> in <cell line: 5>()
          3 b = torch.rand(3, 2)
          4 
    ----> 5 print(a * b)
          6 
          7 #Bu hata, tensörlerin boyutlarının çarpma işlemi için uyuşmaması nedeniyle oluşur.


    RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1


### Tensör Yayılımı (Broadcasting)
Genel olarak, farklı şekillere sahip tensörler doğrudan işleme alınamaz, hatta eleman sayıları eşit olsa bile. Ama Broadcasting uygulayarak yapılabilir.



```python
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)   # (1,4) tensörü 2 ile çarpılır ve (2,4) olarak genişletilir

print(rand)
print(doubled)
```

    tensor([[0.2024, 0.5731, 0.7191, 0.4067],
            [0.7301, 0.6276, 0.7357, 0.0381]])
    tensor([[0.4049, 1.1461, 1.4382, 0.8134],
            [1.4602, 1.2551, 1.4715, 0.0762]])


### Broadcasting Kuralları
Bir tensörle işlem yapmak için şu kurallar geçerli olmalıdır:

1. Her tensör en az bir boyuta sahip olmalıdır (boş tensörler olmaz).
2. Boyutlar sağdan sola karşılaştırılır:Boyutlar eşitse → işlem yapılabilir. Bir boyut 1 ise → genişletilebilir. Bir tensörde olmayan bir boyut varsa → küçük tensör bu boyuta genişletilir.
3. Aynı boyuta sahip tensörler zaten otomatik olarak işlem görebilir.

Broadcasting işlemleri PyTorch'ta tensörleri boyutlarını manuel olarak değiştirmeden kullanmamıza olanak tanır.


```python
a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) #  (4,3,2) ile (3,2) → 1. boyut eksik, genişletilir
print(b)

c = a * torch.rand(   3, 1) # 4,3,2) ile (3,1) → 3. boyut 1, genişletilir
print(c)

d = a * torch.rand(   1, 2) # (4,3,2) ile (1,2) → 2. boyut 1, genişletilir
print(d)
```

    tensor([[[0.2138, 0.5395],
             [0.3686, 0.4007],
             [0.7220, 0.8217]],
    
            [[0.2138, 0.5395],
             [0.3686, 0.4007],
             [0.7220, 0.8217]],
    
            [[0.2138, 0.5395],
             [0.3686, 0.4007],
             [0.7220, 0.8217]],
    
            [[0.2138, 0.5395],
             [0.3686, 0.4007],
             [0.7220, 0.8217]]])
    tensor([[[0.2612, 0.2612],
             [0.7375, 0.7375],
             [0.8328, 0.8328]],
    
            [[0.2612, 0.2612],
             [0.7375, 0.7375],
             [0.8328, 0.8328]],
    
            [[0.2612, 0.2612],
             [0.7375, 0.7375],
             [0.8328, 0.8328]],
    
            [[0.2612, 0.2612],
             [0.7375, 0.7375],
             [0.8328, 0.8328]]])
    tensor([[[0.8444, 0.2941],
             [0.8444, 0.2941],
             [0.8444, 0.2941]],
    
            [[0.8444, 0.2941],
             [0.8444, 0.2941],
             [0.8444, 0.2941]],
    
            [[0.8444, 0.2941],
             [0.8444, 0.2941],
             [0.8444, 0.2941]],
    
            [[0.8444, 0.2941],
             [0.8444, 0.2941],
             [0.8444, 0.2941]]])


Look closely at the values of each tensor above:
* The multiplication operation that created `b` was broadcast over every "layer" of `a`.
* For `c`, the operation was broadcast over ever layer and row of `a` - every 3-element column is identical.
* For `d`, we switched it around - now every *row* is identical, across layers and columns.

For more information on broadcasting, see the [PyTorch documentation](https://pytorch.org/docs/stable/notes/broadcasting.html) on the topic.

Here are some examples of attempts at broadcasting that will fail:

**Note: The following cell throws a run-time error. This is intentional.**


```python
# Aşağıdaki uygulamalar başarısız olur ve çalışma zamanı hatası (RuntimeError) verir.

a =     torch.ones(4, 3, 2)

b = a * torch.rand(4, 3)    # HATA: Son boyutlar eşleşmeli!

c = a * torch.rand(   2, 3) # 3. ve 2. boyutlar farklı!

d = a * torch.rand((0, ))   # Boş tensör ile broadcasting yapılamaz!
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-55-89583052045a> in <cell line: 5>()
          3 a =     torch.ones(4, 3, 2)
          4 
    ----> 5 b = a * torch.rand(4, 3)    # HATA: Son boyutlar eşleşmeli!
          6 
          7 c = a * torch.rand(   2, 3) # 3. ve 2. boyutlar farklı!


    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2


### PyTorch ile Daha Fazla Matematiksel İşlem
PyTorch tensörler üzerinde 300'den fazla işlem yapabilir.


```python
# common functions
a = torch.rand(2, 4) * 2 - 1    # [-1, 1] aralığında rastgele tensör
print('Common functions:')
print(torch.abs(a))    # Mutlak değer
print(torch.ceil(a))   # Yukarı yuvarlama
print(torch.floor(a))  # Aşağı yuvarlama
print(torch.clamp(a, -0.5, 0.5))    # Belirli aralığa sıkıştırma

# trigonometric functions and their inverses
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# bitwise operations
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))    # Bit düzeyinde XOR işlemi

# comparisons:
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # (1,2) boyutundaki tensör, (2,2) boyutuna genişletilecek (broadcasting)!
print(torch.eq(d, e)) #  Eleman bazında karşılaştırma yapar

# reductions:
print('\nReduction ops:')
print(torch.max(d))        # Maksimum değeri döndürür
print(torch.max(d).item()) # Tek değerli tensörden sayıyı çeker
print(torch.mean(d))       # ortalama
print(torch.std(d))        # standard sapma
print(torch.prod(d))       # sayıların çarpımı
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # Benzersiz elemanları döndürür

# vector and linear algebra operations
v1 = torch.tensor([1., 0., 0.])         # x birim vektör
v2 = torch.tensor([0., 1., 0.])        # x birim vektör
m1 = torch.rand(2, 2)                   # random matrix
m2 = torch.tensor([[3., 0.], [0., 3.]]) # 3 katlı birim matris

print('\nVectors & Matrices:')
print(torch.cross(v2, v1)) # # Çapraz çarpım (cross product) (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)  # Matris çarpımı
print(m3)                
print(torch.svd(m3))       #(Tekil değer ayrışımı)
```

    Common functions:
    tensor([[0.8447, 0.1992, 0.9755, 0.9295],
            [0.8190, 0.1029, 0.7480, 0.4949]])
    tensor([[-0., -0., 1., -0.],
            [-0., -0., 1., -0.]])
    tensor([[-1., -1.,  0., -1.],
            [-1., -1.,  0., -1.]])
    tensor([[-0.5000, -0.1992,  0.5000, -0.5000],
            [-0.5000, -0.1029,  0.5000, -0.4949]])
    
    Sine and arcsine:
    tensor([0.0000, 0.7854, 1.5708, 2.3562])
    tensor([0.0000, 0.7071, 1.0000, 0.7071])
    tensor([0.0000, 0.7854, 1.5708, 0.7854])
    
    Bitwise XOR:
    tensor([3, 2, 1])
    
    Broadcasted, element-wise equality comparison:
    tensor([[ True, False],
            [False, False]])
    
    Reduction ops:
    tensor(4.)
    4.0
    tensor(2.5000)
    tensor(1.2910)
    tensor(24.)
    tensor([1, 2])
    
    Vectors & Matrices:
    tensor([ 0.,  0., -1.])
    tensor([[0.6923, 0.7545],
            [0.7746, 0.2330]])
    tensor([[2.0769, 2.2636],
            [2.3237, 0.6990]])
    torch.return_types.svd(
    U=tensor([[-0.7959, -0.6054],
            [-0.6054,  0.7959]]),
    S=tensor([3.7831, 1.0066]),
    V=tensor([[-0.8088,  0.5881],
            [-0.5881, -0.8088]]))


    <ipython-input-56-723b464fcb2b>:46: UserWarning: Using torch.cross without specifying the dim arg is deprecated.
    Please either pass the dim explicitly or simply use torch.linalg.cross.
    The default value of dim will change to agree with that of linalg.cross in a future release. (Triggered internally at ../aten/src/ATen/native/Cross.cpp:62.)
      print(torch.cross(v2, v1)) # # Çapraz çarpım (cross product) (v1 x v2 == -v2 x v1)


### Tensörleri Yerinde (In-Place) Değiştirme
Çoğu ikili işlem (binary operation) yeni bir tensör döndürür.

Geçici hesaplamalarda ara değerleri saklamak istemiyorsak ve Bellek verimliliğini artırmak için uygulanır.

PyTorch'ta birçok matematiksel fonksiyonun (`_`) ekli bir versiyonu vardır ve bu fonksiyonlar tensörü yerinde değiştirir.


```python
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # Yeni bir tensör oluşturur
print(a)              # a değişmez!

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # ın-place işlemi
print(b)              # b değişti
```

    a:
    tensor([0.0000, 0.7854, 1.5708, 2.3562])
    tensor([0.0000, 0.7071, 1.0000, 0.7071])
    tensor([0.0000, 0.7854, 1.5708, 2.3562])
    
    b:
    tensor([0.0000, 0.7854, 1.5708, 2.3562])
    tensor([0.0000, 0.7071, 1.0000, 0.7071])
    tensor([0.0000, 0.7071, 1.0000, 0.7071])


Aritmetik İşlemler İçin Benzer Davranış Gösteren Fonksiyonlar:


```python
a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)
```

    Before:
    tensor([[1., 1.],
            [1., 1.]])
    tensor([[0.8441, 0.9004],
            [0.3995, 0.6324]])
    
    After adding:
    tensor([[1.8441, 1.9004],
            [1.3995, 1.6324]])
    tensor([[1.8441, 1.9004],
            [1.3995, 1.6324]])
    tensor([[0.8441, 0.9004],
            [0.3995, 0.6324]])
    
    After multiplying
    tensor([[0.7125, 0.8107],
            [0.1596, 0.3999]])
    tensor([[0.7125, 0.8107],
            [0.1596, 0.3999]])


in-place aritmetik işlemler `torch.Tensor` nesnesi üzerinde bir metot olarak tanımlanır. Bunlar PyTorch modülüne (`torch.sin()` gibi) bağlı değildir. 

 *`a.add_(b)`, a tensörünü yerinde değiştirir. Yeni bir tensör oluşturulmaz, a bellekte aynı yerde kalır.*

`out` Argümanı ile Bellek Tahsisinden Kaçınma: 
Bazı PyTorch fonksiyonları, sonuçları mevcut bir tensöre yazmak için out argümanını destekler.
Bu yöntem, yeni bir bellek tahsisi yapmadan hesaplamaları aynı tensör üzerine yazar.


```python
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)                # c'nin içeriği değişti!

assert c is d           
assert id(c), old_id    

torch.rand(2, 2, out=c)
print(c)                # c değişti
assert id(c), old_id    # Bellek adresi hala aynı!
```

    tensor([[0., 0.],
            [0., 0.]])
    tensor([[0.6202, 0.3974],
            [0.4075, 0.6706]])
    tensor([[0.4491, 0.6265],
            [0.9411, 0.4922]])


## Tensör Kopyalama (Copying Tensors)
Python'daki herhangi bir nesne gibi, bir tensörü başka bir değişkene atamak, nesnenin kendisini değil, referansını kopyalar.


```python
a = torch.ones(2, 2)
b = a

a[0][1] = 561  # a tensörünü değiştiriyoruz
print(b)       # böylece b de değişti
```

    tensor([[  1., 561.],
            [  1.,   1.]])


Eğer aynı veriye sahip ancak bağımsız bir kopya oluşturmak istiyorsanız,`clone()` metodunu kullanabilirsiniz:


```python
a = torch.ones(2, 2)
b = a.clone()       # Yeni, bağımsız bir kopya oluşturulur

# Doğrulama: a ve b farklı nesneler mi?
assert b is not a      # Bellekte farklı nesneler
print(torch.eq(a, b))  # ama İçerikleri aynı!
a[0][1] = 561          # a nın bir elemanını değiştirelim...
print(b)               # ...ama b hala değişmedi
```

    tensor([[True, True],
            [True, True]])
    tensor([[1., 1.],
            [1., 1.]])


### `clone()` Kullanırken Dikkat Edilmesi Gerekenler:
clone() metodunu kullanırken kaynak tensörünüzün autograd açık olup olmadığını bilmek önemlidir.
* Eğer kaynak tensör requires_grad=True ile oluşturulmuşsa, clone() edilen tensörde de requires_grad=True olur.
Bu, modelinizin `forward()` geçişinde bir tensörün hem kendisinin hem de klonunun çıktıyı etkilemesi durumunda faydalıdır.

* `detach().clone()` Kullanımı : Eğer klonlanan tensörün autograd geçmişini takip etmesini istemiyorsak, detach() kullanabiliriz:



```python
a = torch.rand(2, 2, requires_grad=True) # turn on autograd
print(a)

b = a.clone()     # clone() edilen tensörde de requires_grad=True olur!
print(b)

c = a.detach().clone()    # a'nın türev takibini kaldır ve sonra kopyala
print(c)

print(a)
```

    tensor([[0.5461, 0.5396],
            [0.3053, 0.1973]], requires_grad=True)
    tensor([[0.5461, 0.5396],
            [0.3053, 0.1973]], grad_fn=<CloneBackward0>)
    tensor([[0.5461, 0.5396],
            [0.3053, 0.1973]])
    tensor([[0.5461, 0.5396],
            [0.3053, 0.1973]], requires_grad=True)


Yukarıdaki işlemlerin açıklaması:

* `requires_grad=True`. argümanı ile a tensörü oluşturuluyor. Bu, tensörün autograd mekanizması tarafından takip edileceği anlamına gelir.
* a Tensörünü `clone()` ile Klonlama (b = a.clone()): bu işlem, a tensörünün bağımsız bir kopyasını oluşturur. Ancak, requires_grad=True olduğu için, b de autograd geçmişini korur. Bu yüzden b, a'nın hesaplama geçmişini takip etmeye devam eder.
* `grad_fn=<CloneBackward> ` ifadesi, b tensörünün türevleme sürecinde takip edileceğini gösterir.
* ` detach()` Kullanımı (c = a.detach().clone()): tensörü autograd bağlamından çıkarı, yani artık autograd hesaplamalarına dahil edilmez. Yani, `c.requires_grad=False` olacak ve grad_fn bilgisi içermeyecek.


## GPU'ya Taşımak (Moving to GPU)

PyTorch'un en büyük avantajlarından biri, CUDA uyumlu Nvidia GPU'lar üzerinde hızlandırılmış hesaplama yapabilmesidir.
CUDA (Compute Unified Device Architecture), Nvidia'nın paralel hesaplama platformudur ve büyük ölçekli tensör işlemlerini hızlandırır.


```python
if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')
```

    We have a GPU!
    The history saving thread hit an unexpected error (OperationalError('attempt to write a readonly database')).History will not be written to the database.


Eğer bir veya daha fazla GPU mevcutsa, verimizi GPU’nun erişebileceği bir belleğe taşımamız gerekir.

* CPU, bilgisayarın RAM’inde işlem yapar.
* GPU’nun kendi özel belleği (VRAM) vardır.
* GPU’da işlem yapmak için veriyi VRAM’e taşımamız gerekir.
* Bu işleme genellikle "veriyi GPU'ya taşımak" denir.



```python
# tensörü doğrudan GPU'da oluşturmak:
if torch.cuda.is_available():
    gpu_rand = torch.rand(2, 2, device='cuda')
    print(gpu_rand)
else:
    print('Sorry, CPU only.')
```

    tensor([[0.3344, 0.2640],
            [0.2119, 0.0582]], device='cuda:0')


## GPU'da Tensör Oluşturma ve Cihaz Yönetimi
Varsayılan olarak, yeni tensörler CPU'da oluşturulur. Bu yüzden, GPU'da bir tensör oluşturmak istiyorsak `device` parametresini belirtmemiz gerekir.
PyTorch, tensörün hangi cihazda olduğunu yazdırdığımızda bize bildirir (eğer CPU'da değilse).

* GPU sayısını sorgulamak için `torch.cuda.device_count()` kullanılabilir. Eğer birden fazla GPU varsa, bunları indeksleyerek belirtebiliriz:
*  kodu CPU veya GPU fark etmeksizin daha sağlam hale getirmek için torch.device kullanmalıyız. Bu yöntem, kodun hem CPU'da hem de GPU'da çalışmasını sağlar.


 


```python
if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)
```

    Device: cuda
    tensor([[0.6815, 0.0556],
            [0.0711, 0.4825]], device='cuda:0')



Eğer bir tensör belirli bir cihazda bulunuyorsa, onu başka bir cihaza `to()` metodu ile taşıyabilirsiniz.


```python
y = torch.rand(2, 2)     # CPU'da tensör oluştur
y = y.to(my_device)      # Daha önce belirlenen cihaza taşı
```

İki veya daha fazla tensörle işlem yapabilmek için tüm tensörlerin aynı cihazda olması gerekir.
```
x = torch.rand(2, 2)
y = torch.rand(2, 2, device='gpu')
z = x + y   # HATA! CPU ve GPU tensörleri birlikte işlenemez
```
* Tüm tensörleri aynı cihaza taşımak için .to() metodunu kullanmalıyız

## Manipulating Tensor Shapes

Bazen tensörün şeklini değiştirmemiz gerekir. Bu, özellikle PyTorch modellerine giriş verirken yaygın bir durumdur.

### Boyut Sayısını Değiştirme 
Tek bir örneği modele verirken boyut eklememiz gerekebilir.
Örneğin, bir modelin 3x226x226 şekline sahip bir görüntü üzerinde çalıştığını düşünelim:
* Tensörümüz: (3, 226, 226), Modelin beklediği giriş: (N, 3, 226, 226)
N, batch (küme) boyutudur ve modele kaç görüntü verdiğimizi belirtir.

Peki tek bir görüntü (örnek) için batch oluşturmak istersek ne yaparız?
* `.unsqueeze()` Kullanmak: .unsqueeze(dim) metodu, belirtilen boyuta yeni bir eksen ekler. Model artık tek bir örneği batch formatında alabilir.
* Eğer batch eksenini kaldırmak istersek, `.squeeze(dim)` kullanabiliriz:


```python
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)
```

    torch.Size([3, 226, 226])
    torch.Size([1, 3, 226, 226])


`unsqueeze()` boyut ekler: `unsqueeze(0)`  tensörün 0. eksenine boyut 1 ekler. Bu sayede tensör, modelin beklediği batch formatına uygun hale gelir.

`squeeze()` ile gereksiz boyutları kaldırma


```python
c = torch.rand(1, 1, 1, 1, 1)
print(c)
```

    tensor([[[[[0.8637]]]]])


Önceki örneğe devam edersek, diyelim ki modelin çıktısı her giriş için 20 elemanlı bir vektör üretiyor. Bu durumda, çıktının şekli (N, 20) olur, burada N giriş batch boyutudur. Eğer tek bir girişimiz varsa, model çıktısı `(1, 20)` olur.

 Peki, bu çıktıyı batch olmadan, sadece 20 elemanlı bir vektör olarak kullanmak istersek ne yaparız?
*squeeze()*  Kullanmak


```python
a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)    # non-batched
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)
```

    torch.Size([1, 20])
    tensor([[0.1895, 0.9874, 0.7688, 0.6143, 0.8682, 0.0899, 0.4232, 0.5541, 0.1231,
             0.6651, 0.7216, 0.8821, 0.9117, 0.7213, 0.9232, 0.9339, 0.5014, 0.8377,
             0.3018, 0.3514]])
    torch.Size([20])
    tensor([0.1895, 0.9874, 0.7688, 0.6143, 0.8682, 0.0899, 0.4232, 0.5541, 0.1231,
            0.6651, 0.7216, 0.8821, 0.9117, 0.7213, 0.9232, 0.9339, 0.5014, 0.8377,
            0.3018, 0.3514])
    torch.Size([2, 2])
    torch.Size([2, 2])



Gördüğünüz gibi, iki boyutlu bir tensör `squeeze()`  ile tek boyutlu hale gelir.
Eğer çıktıyı yazdırırsanız, fazladan `[]`köşeli parantezler olduğunu fark edersiniz.
* `squeeze()`  sadece uzunluğu 1 olan eksenleri kaldırabilir. Tensördeki eleman sayısını değiştirmeden boyutları manipüle eder.

* Bazı durumlarda, `unsqueeze()` kullanarak tensörleri broadcasting için uygun hale getirebiliriz.

```
a =     torch.ones(4, 3, 2)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)    # Çıktı: torch.Size([4, 3, 2])
```


```python
a = torch.ones(4, 3, 2)
b = torch.rand(   3)     #  a * b hata verecektir, çünkü boyutlar broadcasting kurallarına uymuyor.
c = b.unsqueeze(1)       # change to a 2-dimensional tensor,b tensörüne ekstra bir boyut ekleyebiliriz:

print(a * c)            
```

    tensor([[[0.4244, 0.4244],
             [0.2675, 0.2675],
             [0.5692, 0.5692]],
    
            [[0.4244, 0.4244],
             [0.2675, 0.2675],
             [0.5692, 0.5692]],
    
            [[0.4244, 0.4244],
             [0.2675, 0.2675],
             [0.5692, 0.5692]],
    
            [[0.4244, 0.4244],
             [0.2675, 0.2675],
             [0.5692, 0.5692]]])


`squeeze()` ve `unsqueeze()` metodlarının in-place versiyonları da vardır.


```python
batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)

batch_me.unsqueeze_(0)    # 0. boyuta yeni bir batch ekseni ekle
print(batch_me.shape)
```

    torch.Size([3, 226, 226])
    torch.Size([1, 3, 226, 226])


Tensörün şeklini radikal bir şekilde değiştirmek gerekebilir, ancak eleman sayısını ve içeriğini koruyarak. Bu durum, özellikle evrişimsel (convolutional) katman ile linear katman arasındaki geçişte yaygındır.

### `reshape()`Kullanarak Tensörü Tek Boyutlu Hale Getirme:
Evrişimsel bir katmanın çıkışı genellikle (features, width, height) formatındadır, ancak tam bağlantılı (linear) katmanlar genellikle 1D giriş bekler.
Bu yüzden reshape() kullanarak tensörü düzleştirmeliyiz.



```python
output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)    # Tek boyutlu hale getir
print(input1d.shape)

# Aynı işlemi torch modül seviyesinde de yapabiliriz:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)
```

    torch.Size([6, 20, 20])
    torch.Size([2400])
    torch.Size([2400])


* Note: PyTorch reshape() metodunda şeklin bir tuple olmasını bekler.
Parantez (6 * 20 * 20,) kullanarak tuple olduğunu belirtiyoruz.
Eğer ilk argüman olarak direkt reshape() çağırırsak, virgülsüz integer serisi kullanılabilir.

* reshape() çoğu zaman tensörün bellekte yeni bir kopyasını oluşturmaz.
Bunun yerine aynı belleği paylaşan yeni bir görünüm (view) oluşturur. Eğer bağımsız bir kopya istiyorsak, clone() kullanmalıyız.


## NumPy Bridge

PyTorch, NumPy ile uyumlu yayınlama (broadcasting) kurallarına sahip olmanın ötesinde, NumPy dizileri (ndarray) ile doğrudan dönüşüm yapabilir.

Bu, mevcut ML veya bilimsel kodlarınızı PyTorch'a taşımak için büyük bir avantaj sağlar.


```python
import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)   # PyTorch tensörüne çevir
print(pytorch_tensor)
```

    [[1. 1. 1.]
     [1. 1. 1.]]
    tensor([[1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)


PyTorch, NumPy dizisinin şekli ve içeriğini koruyarak bir tensör oluşturur.
Hatta NumPy’nın varsayılan 64-bit float (float64) veri tipini bile korur.

* dönüşüm tam tersi yönde de kolayca yapılabilir.


```python
pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)
```

    tensor([[0.1103, 0.3115, 0.2571],
            [0.9577, 0.3313, 0.4121]])
    [[0.11027098 0.3114581  0.25705016]
     [0.957737   0.33131754 0.4120825 ]]


PyTorch tensörleri ile NumPy dizileri arasında dönüşüm yaparken, iki nesne aynı belleği paylaşır. Yani birinde yapılan değişiklik diğerine de yansır.


```python
numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
```

    tensor([[ 1.,  1.,  1.],
            [ 1., 23.,  1.]], dtype=torch.float64)
    [[ 0.11027098  0.3114581   0.25705016]
     [ 0.957737   17.          0.4120825 ]]


