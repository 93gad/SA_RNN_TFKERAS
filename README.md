# Анализ тональности с использованием Tensorflow и Keras
Этот репозиторий содержит код для анализа тональности текстов на русском языке с использованием библиотек Tensorflow и Keras.

# Требования
- Tensorflow
- Keras
- Numpy
- Pandas
- Matplotlib
- Re


# Данные
Данные, используемые для обучения модели, можно найти по ссылке ниже.

[Google Drive](https://drive.google.com/drive/folders/1XXU66O306ahuMs_X1LQZYSa9kcKU8lS4?usp=share_link)			[![Google_Drive_icon_(2020) (1)](https://user-images.githubusercontent.com/118125931/216775668-dd2e04ed-c06d-4c8e-b186-0e92789c98a3.png)](https://drive.google.com/drive/folders/1XXU66O306ahuMs_X1LQZYSa9kcKU8lS4?usp=share_link)


# Запустить в Google Colab    
[Google Colab](https://colab.research.google.com/drive/1NaYKqfcRwelH-FywcJ0zhbSQppYsMowa?usp=sharing)			[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NaYKqfcRwelH-FywcJ0zhbSQppYsMowa?usp=sharing)


# Алгоритм анализа тональности с использованием RNN



### Загрузка данных

```python
!gdown --id 10v9w9Ss2luAUUuDXWVe25zTfNe_JHUNY
!gdown --id 1ZL95-9w8CxbBsy1XOt71ogGbQ4vtMgm8
```


```python
train = pd.read_csv('train.csv', 
                    header=None, 
                    names=['Review', 'Class'],)
test = pd.read_csv('test.csv', 
                    header=None, 
                    names=['Review', 'Class'],)
```



```python
print("Размер набора для обучения: ", len(x_train_text))
print("Размер набора для тестирования:  ", len(x_test_text))
```
  
### Токенизация

```python
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
```


### Создание модели

```python
model = Sequential()
```

    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, 1558, 8)           80000     

     gru (GRU)                   (None, 1558, 16)          1248      

     dropout (Dropout)           (None, 1558, 16)          0         

     gru_1 (GRU)                 (None, 1558, 8)           624       

     gru_2 (GRU)                 (None, 4)                 168       

     dense (Dense)               (None, 1)                 5         

    =================================================================
    Total params: 82,045
    Trainable params: 82,045
    Non-trainable params: 0
    _________________________________________________________________



### Обучение

```python
%%time
history=model.fit(x_train_pad, y_train,
          validation_split=0.05, epochs=3, batch_size=50)

# one epoch = один прямой проход и один обратный проход всех обучающих примеров
# batch size = количество обучающих примеров за один прямой / обратный проход
```

    Epoch 1/3
    126/126 [==============================] - 27s 126ms/step - loss: 0.5033 - accuracy: 0.8213 - val_loss: 0.4302 - val_accuracy: 0.8399
    Epoch 2/3
    126/126 [==============================] - 14s 113ms/step - loss: 0.3899 - accuracy: 0.8390 - val_loss: 0.3271 - val_accuracy: 0.8731
    Epoch 3/3
    126/126 [==============================] - 14s 113ms/step - loss: 0.2405 - accuracy: 0.9099 - val_loss: 0.3293 - val_accuracy: 0.8701
    CPU times: user 52 s, sys: 1.24 s, total: 53.2 s
    Wall time: 1min 30s

       
        


```python
positive_review='''Прогнозом поделился директор Департамента гидрологии РГП «Казгидромет» Адель Ахметов. Во время брифинга Региональной службы коммуникаций Алматы он подчеркнул, что в мегаполисе ожидается ранняя весна.

«В первой декаде марта ожидается повышение температуры воздуха до +2+7°С, днем до +10+15°С. В среднем температура воздуха ожидается выше климатической нормы на 1°С. Более детализированный прогноз на март 2023 года по городу Алматы будет выпущен 15 февраля и будет уточняться декадными и недельными прогнозами», - резюмировал Ахметов.'''

negative_review='''32-летнюю жительницу подозревают в продаже своих детей в Атырау, сообщает издание «Ақ Жайық»

Как пишет издание, дело касается двух малышей в возрасте одного года и двух лет.

Полицейские открыли досудебное расследование по статье УК РК «Торговля несовершеннолетними».'''

text=[positive_review,negative_review]
```

### Проверка на неизвестных данных


```python
tokens = tokenizer.texts_to_sequences(text) # we need to tokenize
```


```python
tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating='pre')
```


```python
tokens_pad.shape
```




    (2, 1558)




```python
pos_review=model.predict(tokens_pad)[0]
neg_review=model.predict(tokens_pad)[1]
```


```python
print('Позитивный текст с оценкой {}'.format(a[0]*100))

```

>>Позитивный текст с оценкой 92.99412965774536



    


```python
print('Негативный текст с оценкой {}'.format(b[0]*100))
```
>>Негативный текст с оценкой 8.072149008512497


