# Сопоставимость протоколов: KuaiRand Protocol B vs published benchmarks

## A. Наш Protocol B

Источник локальных чисел: [kuairand_protocol_b_data_report.md](../../reports/kuairand_protocol_b_data_report.md) и [protocol_b_manifest.json](../../outputs/data/protocol_b_manifest.json).

| Поле | Значение |
| --- | --- |
| users | 23,951 |
| items | 7,111 |
| interactions | 1,134,420 |
| train | 1,086,518 |
| validation | 23,951 |
| test | 23,951 |
| filtering | iterative 5-core по user и item |
| ordering | chronological, with source row tie-break |
| split | leave-one-out: penultimate validation, last test |
| max sequence length | 50 |
| evaluation | full ranking over Protocol B item universe |
| fingerprint | `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7` |

## B. SSD4Rec protocol and versioning

Primary sources: SSD4Rec paper v1, current SSD4Rec arXiv v2, official repository and official config.

- SSD4Rec arXiv v1 Table 4 is the historical reproduction target for this project: https://arxiv.org/html/2409.01192v1.
- SSD4Rec current arXiv v2 Table 4 is an updated published paper version: https://arxiv.org/pdf/2409.01192.
- These two versions are not treated as an error pair. They are two published versions and therefore are represented as separate `benchmark_source` rows.
- Official repository: https://github.com/ZhangYifeng1995/SSD4Rec.
- Config snapshot: https://raw.githubusercontent.com/ZhangYifeng1995/SSD4Rec/bdbfe5193f3a6697bb6ee0699ab43386d80c6198/config.yaml.
- The checked config supports protocol compatibility: `MAX_ITEM_LIST_LENGTH=50`, user/item `5-core`, `loss_type='CE'`, `train_neg_sample_args=~`, `metrics=['NDCG','MRR','Hit']`, `valid_metric=NDCG@10`, `topk=[10,20]`.

## C. TiM4Rec protocol

Primary sources: TiM4Rec paper, official repository and official KuaiRand config.

- TiM4Rec Table 3 is represented as its own `benchmark_source`: https://arxiv.org/html/2409.16182v3.
- Table 3 includes Caser, GRU4Rec, SASRec, BERT4Rec, TiSASRec, LRURec, Mamba4Rec, SSD4Rec* and TiM4Rec for KuaiRand at @10/@20/@50.
- `SSD4Rec*` in TiM4Rec Table 3 is the TiM4Rec authors own replicated variant. It must not be read as the original SSD4Rec published row.
- Official repository: https://github.com/AlwaysFHao/TiM4Rec.
- Config snapshot: https://raw.githubusercontent.com/AlwaysFHao/TiM4Rec/8d4a6cea6a035c249a7a13999166ba41e8924abe/config/config4kuai_64d.yaml.
- The checked config supports protocol compatibility: `dataset=kuairand`, `MAX_ITEM_LIST_LENGTH=50`, user/item `5-core`, `train_neg_sample_args=~`, `metrics=['Hit','NDCG','MRR']`, `valid_metric=NDCG@10`, `topk=[10,20,50]`.

## D. Почему STRONGLY_COMPARABLE, а не EXACT

`STRONGLY_COMPARABLE` означает, что задокументированы та же public dataset family, те же 5-core/leave-one-out/full-ranking sequential benchmark semantics и совпадающие RecBole-style configuration knobs. Это не означает byte-level equivalence.

Мы не заявляем `EXACT`, потому что papers не публикуют immutable processed file hashes, exact split row ids или полный byte-level fingerprint подготовленных KuaiRand files. У нашего Protocol B есть frozen local fingerprint и hashes, но это наши artifacts, а не proof of identity with the authors processed copy.

## E. Исключённые / несопоставимые работы

| Study | Evidence | Decision | Reason |
| --- | --- | --- | --- |
| FuXi-Linear 2026 | https://arxiv.org/pdf/2602.23671 | excluded from canonical Protocol B table | Uses Kuairand-27K: about 27,284 users / 131,090 items / 97,010,279 interactions / average length 3555.57, plus item-frequency filtering with >100 item interactions. Это не наш Protocol B fingerprint и не наш regime. |
