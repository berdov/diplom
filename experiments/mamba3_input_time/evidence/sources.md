# Проверенные источники и маскирование

Проверено 18.09.2026. Реализация адаптера собственная по формулам в [README](../README.md),
без vendoring стороннего кода и без запуска чужого training stack.

## QuITE

[Статья, v3](https://arxiv.org/html/2605.28166v3),
[pinned tree](https://github.com/Meaningfull9502/QuITE/tree/5bc9b00a6a9b07ffb48f2c0548cf9c662c087e81).
Версия `5bc9b00a6a9b07ffb48f2c0548cf9c662c087e81`. Прочитаны:

- [models/embeddings/quite.py](https://github.com/Meaningfull9502/QuITE/blob/5bc9b00a6a9b07ffb48f2c0548cf9c662c087e81/models/embeddings/quite.py): learned variable/patch queries, итоговый query token вместо сохранения всех L событий.
- [models/modules.py](https://github.com/Meaningfull9502/QuITE/blob/5bc9b00a6a9b07ffb48f2c0548cf9c662c087e81/models/modules.py): attention, mask broadcasting, residual/LayerNorm.
- [models/embeddings/_base.py](https://github.com/Meaningfull9502/QuITE/blob/5bc9b00a6a9b07ffb48f2c0548cf9c662c087e81/models/embeddings/_base.py): learned harmonic timestamp embedding.
- [models/quite.py](https://github.com/Meaningfull9502/QuITE/blob/5bc9b00a6a9b07ffb48f2c0548cf9c662c087e81/models/quite.py): wrapper для forecasting/classification, переменных и patches.

В pinned recursive Git tree отсутствует LICENSE/COPYING; raw LICENSE возвращает 404.
Badge не принят как разрешение на копирование. Классы QuITE не копировались.

Путь variate/patch передаёт mask `[B',L,1]`, где B' объединяет batch и variable/patch.
Следующий unsqueeze даёт `[B',1,L,1]` при scores `[B',H,L,L]`: маскируются query rows,
не key columns. Для query с mask=1 masked observation остаётся доступным key.
Собственный детерминированный тест выражения broadcasting: Q=K=0, mask=[1,1,0],
values=[(0,0),(0,0),(1,2)]. Замена последнего value на (10,20) меняет
attention output действительного query на **(3,6)**. Правильная key mask даёт разницу **(0,0)**.
Дополнительный четырёхмерный fixture с неколлинеарными values подтверждает изменение
query embedding и после residual + LayerNorm; правильная key mask сохраняет его точно.
Это локальное воспроизведение данного пути, не опровержение всей статьи/её результатов.
[Тест](../tests/test_adapter.py) `test_quite_pinned_broadcast_reproduction` и trained-like
regression проверяют также независимость нового адаптера от masked values и запрещённого suffix.

## TiSASRec и границы

[TiSASRec, WSDM 2020](https://doi.org/10.1145/3336191.3371786),
[авторский репозиторий](https://github.com/JiachengLi1995/TiSASRec),
[model.py](https://github.com/JiachengLi1995/TiSASRec/blob/master/model.py),
[modules.py](https://github.com/JiachengLi1995/TiSASRec/blob/master/modules.py).
Проверены pairwise interval embeddings для K/V и absolute-position K/V в causal attention.
У нас непрерывный log-interval MLP добавляет per-head scalar bias к logits перед Mamba3,
без новых value/position embeddings. Временной attention в рекомендациях уже существует;
наш пилот не exact TiSASRec reproduction и не заявка на первое применение.

[Mantis](https://github.com/vfeofanov/mantis) является pretrained foundation model для time-series
classification; её интеграция меняла бы модель, представление и постановку этого ограниченного
scratch recommender пилота. Ни библиотека, ни веса не устанавливались/не загружались.
