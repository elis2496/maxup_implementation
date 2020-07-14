### Имплементация MaxUp

Статья - https://arxiv.org/pdf/2002.09024v1.pdf

За основу взята модель ResNet34

#### Необходимые пакеты

Python 3.7.7

```pip install -r requirements.txt```

#### Обучение модели

```python train.py```

Аргументы:

* ```cutmix``` - использовать ли CutMix;
* ```m``` - число аугментаций CutMix для Maxup;
* ```device``` - использовать CPU или GPU;
* ```epochs``` - число эпох;
* ```pretrained_weights``` - путь к весам предобученной модели.

#### Оценка качества

```python eval.py```

Аргументы:

* ```weights``` - путь к весам модели;
* ```device``` - использовать CPU или GPU.

#### Результаты 

Сравнительная оценка качества:

|  | Accuracy, % |
| :---: | :---: |
| Resnet34 | 98.5 |
| Resnet34 + CutMix| 97.49 |
| Resnet34 + MaxUp+CutMix, m=4 | 96.83 |
| Resnet34 + MaxUp+CutMix, m=4, fine-tuned| 97.02 |

Веса моделей:
 
 * Resnet34: ```./result/Base_exp/weights.pth```

 * Resnet34 + CutMix (m=1): ```./result/Cutmix_exp/weights.pth```

 * Resnet34 + MaxUp+CutMix, m=4: ```./result/Cutmix_maxup_4_exp/weights.pth```

 * Resnet34 + MaxUp+CutMix, m=4 (fine-tuned): ```./result/Cutmix_maxup_4_exp/weights.pth```