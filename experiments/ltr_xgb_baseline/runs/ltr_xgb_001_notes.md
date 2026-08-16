# ltr_xgb_001 notes

## Hypothesis

Простой LambdaMART на popularity/history features должен превосходить Random и желательно MostPopular.

## Setup

- Protocol: `kuairand_protocol_b`.
- Candidate protocol: `protocol_b_1pos_100neg_seed42`.
- Query: `user_id`; positives: один target item; negatives/query: `100`.
- Training design: один query на пользователя, positive = последний item из train, context = train history без этого target.
- Raw `user_id` и `item_id` не использовались как числовые признаки модели.
- Remote artifact path: `/home/daryumin/iberdov/diplom/experiments/ltr_xgb_baseline/ltr_xgb_001`.
- XGBoost trees trained: `31`; best_iteration: `0`.

## Result

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

## Comparison

- Random и MostPopular посчитаны тем же evaluation pipeline на тех же fixed candidates.
- HR и Recall равны по всем K, потому что в каждом query ровно один relevant item.

## Observations

- XGBoost HR@10 на test: `0.494802`.
- MostPopular HR@10 на test: `0.495637`.
- Random HR@10 на test: `0.100246`.
- XGBoost превосходит Random по HR@10.
- XGBoost не превосходит MostPopular по HR@10.

## Problems

- XGBoost отсутствовал в `.conda` на cHARISMa; установлен `xgboost` из внутреннего PyPI proxy кластера.
- Для queries, где target item уже встречался в history, target исключался из context перед построением признаков.

## Next step

Добавить более содержательные leakage-safe item/user metadata features или сравнить с RecBole sequential baseline на том же fixed candidate protocol.
