# ltr_xgb_001 notes

## Важно про evaluation protocol

`ltr_xgb_001` является валидным sanity/exploratory experiment, но не является Protocol B full-ranking benchmark.

- `evaluation_protocol`: `B_split_sampled_100_candidates`;
- training candidate protocol: `sampled_100`;
- validation/test candidate protocol: `sampled_100`;
- в каждом validation/test query использовались 1 positive + 100 sampled negatives.

Метрики `ltr_xgb_001` не сопоставимы с опубликованными SSD4Rec/TiM4Rec Protocol B, потому что там используется full-ranking evaluation. Сами метрики ниже сохранены без изменений.

Sanity observation: при 101 candidate Random HR@10 теоретически около `10 / 101 = 0.099010`. Фактически в 001 Random HR@10 был `0.100539` на validation и `0.100246` на test. Это объясняет высокие абсолютные значения sampled-eval метрик и не означает, что XGBoost превосходит SOTA.

## Гипотеза

Простой LambdaMART на popularity/history features должен превосходить Random и желательно MostPopular.

## Setup

- Protocol: `kuairand_protocol_b`.
- Candidate protocol: `protocol_b_1pos_100neg_seed42`.
- Evaluation protocol: `B_split_sampled_100_candidates`.
- Query: `user_id`; positives: один target item; negatives/query: `100`.
- Training design: один query на пользователя, positive = последний item из train, context = train history без этого target.
- Raw `user_id` и `item_id` не использовались как числовые признаки модели.
- Remote artifact path: `/home/daryumin/iberdov/diplom/experiments/ltr_xgb_baseline/ltr_xgb_001`.
- XGBoost trees trained: `31`; best_iteration: `0`.

## Результат

Validation:

| Model | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.049601 | 0.100539 | 0.196610 | 0.497223 | 0.029255 | 0.045511 | 0.069467 | 0.128156 | 0.049601 | 0.100539 | 0.196610 | 0.497223 |
| MostPopular | 0.353263 | 0.508246 | 0.703353 | 0.921047 | 0.243150 | 0.292946 | 0.342296 | 0.385927 | 0.353263 | 0.508246 | 0.703353 | 0.921047 |
| XGBoost LambdaMART | 0.351342 | 0.508162 | 0.701015 | 0.917415 | 0.241822 | 0.292215 | 0.340996 | 0.384409 | 0.351342 | 0.508162 | 0.701015 | 0.917415 |

Test:

| Model | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.049685 | 0.100246 | 0.194856 | 0.493549 | 0.029187 | 0.045272 | 0.068885 | 0.127278 | 0.049685 | 0.100246 | 0.194856 | 0.493549 |
| MostPopular | 0.342407 | 0.495637 | 0.687278 | 0.916872 | 0.236618 | 0.285787 | 0.334187 | 0.380258 | 0.342407 | 0.495637 | 0.687278 | 0.916872 |
| XGBoost LambdaMART | 0.342282 | 0.494802 | 0.686485 | 0.913240 | 0.236306 | 0.285290 | 0.333690 | 0.379220 | 0.342282 | 0.494802 | 0.686485 | 0.913240 |

## Сравнение

- Random и MostPopular посчитаны тем же evaluation pipeline на тех же fixed candidates.
- HR и Recall равны по всем K, потому что в каждом query ровно один relevant item.
- Сравнение допустимо только внутри sampled-100 protocol, а не с full-ranking literature numbers.

## Наблюдения

- XGBoost HR@10 на test: `0.494802`.
- MostPopular HR@10 на test: `0.495637`.
- Random HR@10 на test: `0.100246`.
- XGBoost превосходит Random по HR@10 внутри sampled-100 evaluation.
- XGBoost не превосходит MostPopular по HR@10.

## Проблемы

- XGBoost отсутствовал в `.conda` на cHARISMa; установлен `xgboost` из внутреннего PyPI proxy кластера.
- Для queries, где target item уже встречался в history, target исключался из context перед построением признаков.

## Следующий шаг

Выполнить `ltr_xgb_002`: оставить sampled-100 training, но пересчитать validation/test как full-ranking по item universe Protocol B.
