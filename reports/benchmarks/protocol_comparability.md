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

## B. Протокол SSD4Rec

Primary sources: SSD4Rec paper v1, official repository and official config.

- Источник paper results: https://arxiv.org/html/2409.01192v1.
- Official repository: https://github.com/ZhangYifeng1995/SSD4Rec.
- Config snapshot: https://raw.githubusercontent.com/ZhangYifeng1995/SSD4Rec/bdbfe5193f3a6697bb6ee0699ab43386d80c6198/config.yaml.
- Paper фиксирует leave-one-out training/validation/test partition и reports KuaiRand in Table 4.
- Official config подтверждает KuaiRand-compatible settings для этого audit: `MAX_ITEM_LIST_LENGTH=50`, user/item `5-core`, `loss_type='CE'`, `train_neg_sample_args=~`, `metrics=['NDCG','MRR','Hit']`, `valid_metric=NDCG@10`, `topk=[10,20]`.
- Repository/paper setup основан на RecBole/PyTorch; full-ranking semantics согласованы с `train_neg_sample_args=~` и нашим RecBole-style Protocol B evaluation, но exact processed split hashes не опубликованы.

## C. Протокол TiM4Rec

Primary sources: TiM4Rec paper, official repository and official KuaiRand config.

- Источник paper results: https://arxiv.org/html/2409.16182v3.
- Official repository: https://github.com/AlwaysFHao/TiM4Rec.
- Config snapshot: https://raw.githubusercontent.com/AlwaysFHao/TiM4Rec/8d4a6cea6a035c249a7a13999166ba41e8924abe/config/config4kuai_64d.yaml.
- Repo README states, что KuaiRand data provided by the SSD4Rec author, и lists RecBole 1.2.0 in the environment.
- `config/config4kuai_64d.yaml` подтверждает `dataset=kuairand`, `MAX_ITEM_LIST_LENGTH=50`, user/item `5-core`, `train_neg_sample_args=~`, `metrics=['Hit','NDCG','MRR']`, `valid_metric=NDCG@10`, `topk=[10,20,50]`.
- Paper reports Table 3 with KuaiRand HR/NDCG/MRR @10/@20/@50.

## D. Почему STRONGLY_COMPARABLE, а не EXACT

`STRONGLY_COMPARABLE` означает, что задокументированы та же public dataset family, те же 5-core/leave-one-out/full-ranking sequential benchmark semantics и совпадающие RecBole-style configuration knobs. Это не означает byte-level equivalence.

Мы не заявляем `EXACT`, потому что papers не публикуют immutable processed file hashes, exact split row ids или полный byte-level fingerprint подготовленных KuaiRand files. У нашего Protocol B есть frozen local fingerprint и hashes, но это наши artifacts, а не proof of identity with the authors' private processed copy.

## E. Исключённые / несопоставимые работы

| Study | Evidence | Decision | Reason |
| --- | --- | --- | --- |
| FuXi-Linear 2026 | https://arxiv.org/pdf/2602.23671 | excluded from canonical table | Uses Kuairand-27K: about 27,284 users / 131,090 items / 97,010,279 interactions / average length 3555.57, plus item-frequency filtering with >100 item interactions. Это не наш Protocol B fingerprint и не наш regime. |
| TiM4Rec Table 3 LRURec/TiSASRec | https://arxiv.org/html/2409.16182v3 | not promoted to canonical table | Table имеет KuaiRand rows, но official logs не найдены в текущем official repo checkout; prompt требовал table/page плюс official logs / same-pipeline confirmation до добавления optional rows. |
