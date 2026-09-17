# Дипломный проект

Последовательная рекомендация на **KuaiRand: хронологический leave-one-out,
оценка по полному каталогу**. Primary-only Mamba3 и временные механизмы:
vanilla, shared RT, decay_only, scan_only, separate.

## Результаты и воспроизводимость

- **[Опубликованные benchmarks и наши TEST](reports/PAPER_RESULTS.md)**:
  shared RT TEST NDCG@10 0.0613 против paper TiM4Rec 0.0611
  (+0.0002; +0.3273%). Сопоставимость всех деталей не подтверждена;
  таблица показывает также проигрыши @20/@50 и ограничения одного seed.
- **[Наши эксперименты и абляции](reports/RESULTS.md)**: TEST отдельно от VALID.
- **[Полный реестр](experiments/results.csv)**: исходные run IDs, commits и JSON.
- **[Experimental setup](reports/EVALUATION_SETUP.md)**: данные, split, masks, timestamps.
- **[Временные механизмы: таблицы и графики](reports/MAMBA3_TIME_MECHANISMS_RESULTS.md)**:
  separate 0.0633: best observed single-seed VALID; confirmation pending.

23 951 пользователей, 7 111 items, 1 134 420 взаимодействий; история до 50.
Данные и [manifest](outputs/data/protocol_b_manifest.json) не менялись.
[Frozen vanilla Mamba3Rec](experiments/mamba3_baseline/README.md):
primary-only, non-time-aware baseline, TEST NDCG@10 **0.0590**.

## Завершённые диагностические этапы

[MTL/MOO study](reports/MTL_MOO_STUDY.md) не выбран основой proposed method.
EPO: лучший observed MOO representative; screening 16 subsets дал до
+0.0007 VALID NDCG@10 на одном seed. Исторические/отрицательные результаты
сохранены в сводке и реестре. Proto-Mamba3 KMeans: VALID 0.0583 против vanilla
0.0584, TEST не запускался. Random-prototype controls не включены в main.
