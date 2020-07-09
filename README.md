### Имплементация MaxUp

Статья - https://arxiv.org/pdf/2002.09024v1.pdf

За основу взята модель ResNet34

#### Необходимые пакеты

Python 3.7.7

```pip install -r requirements.txt```

#### Обучение модели

```python train.py```

Аргументы:

* ```m``` - число аугментаций Cutout для Maxup;
* ```device``` - использовать CPU или GPU;
* ```result_path``` - директория для сохранения весов модели.

#### Оценка качества

```python eval.py```

Аргументы:

* ```weights``` - путь к весам модели;
* ```device``` - использовать CPU или GPU.

#### Результаты 

Сравнительная оценка качества:

|  | Accuracy, % |
| :---: | :---: |
| Resnet34 |  |
| Resnet34 + CutMix|  |
| Resnet34 + MaxUp+CutMix, m=4 |  |


Веса моделей:
 
 * Resnet34:

 * Resnet34 + CutMix (m=1):

 * Resnet34 + MaxUp+CutMix, m=4:

