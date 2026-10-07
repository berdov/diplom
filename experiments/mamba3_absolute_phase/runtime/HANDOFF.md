# Абсолютная фаза: пилот опубликован, confirmation отправлена

Phase: SUBMITTED_PENDING. **2026-10-07 15:08:45 MSK:** PENDING (Priority), node UNKNOWN, elapsed 00:00:00. Scientific fits started UNKNOWN, completed UNKNOWN/12. GPU gate UNKNOWN; smoke UNKNOWN. Next explicit poll no earlier than 2026-10-07 15:18:45 MSK (2026-10-07T12:18:45.702145+00:00). No background process.

## Текущая confirmation, 07.10.2026

Execution: `5282984f078e7d06bbd6ef66249bd7fbf375d3dd`.
Source: `4da0d232b6a73757dbe6065d7d4190d6434cc6f31ca92453ba09fa3c2d2ac764`.
Отдельный `source_manifest_confirmation.json`: те же447 paths, только4
reviewed fixture/config/provenance changes; остальные443 побайтно прежние.
Исходные pilot manifest, raw evidence и план не изменены. Lineage и decision
в runtime связаны SHA с login/reservation. Их байты после submit НЕ МЕНЯТЬ.
CPU100/100 и отдельный no-Git100/100 PASS; 29 stdlib regression tests PASS.
TRAIN coverage PASS: `7d31362f9a63c2ee0674a1b1b0c549c5df59f9b9070133d775cea7f70806a07c`.
Контрольные данные, модель, policy, периоды и fit rules не менялись.

Canonical cluster HEAD соответствует execution; tracked clean;21 известных
untracked файлов stage_confirmation/target_combination_analysis сохранены
побайтно. Checkout под job НЕ МЕНЯТЬ. SSH работает с явно выбранным уже
загруженным ключом: `ssh -i ~/.ssh/hse_karizma hse-karizma`; SSH config не меняли.

Budget после submit:2 jobs,14h requested GPU;3 pilot fits complete,
до12 confirmation fits. TEST0. Seeds2027–2030, baseline→relative→absolute.
Frozen pilot rule absolute>=обоих paired controls выполнено.
Пилот опубликован ordinary fast-forward в main `c414c2f`, registry128.
Сведения о подготовке ниже исторические; стадия ожидания submit уже пройдена.

Следующий шаг: один compact poll через runtime/capture_status.py с
`--stage confirmation --ssh-identity ~/.ssh/hse_karizma`, далее >=600s.
Очередь максимум4h в сессии. Terminal: preserve_terminal.py, stdlib audit,
four-new/all-five отдельно, TeX+primary SVG, registry140 только при12/12 PASS.
Не повторять started fits. Неполную серию не объявлять COMPLETE.

Запрос: `/Users/berdov/.codex/attachments/867dda07-c9ee-42e7-a7ff-e83250722c3e/Вставленный текст.txt`.
Репозиторий: `/Users/berdov/diplom`, cluster `/home/daryumin/iberdov/diplom`,
SSH `hse-karizma`, ветка `exp/mamba3-absolute-phase`.
Исходный main: `de2cddf271bd4c5ccaa4693552f005799a171526`.

## Пилот

Execution: `11913ff8fe99aa28f124d667e2ae3aad14465098`.
Source hash: `4c404906e61121f967307e8c3e97a63e9a476711202a4a5baa005ef9899574c6`,
447 frozen files. Пилотный source_manifest.json не менять.
Preservation и audit опубликованы в commit `f3aad61`.
Audit SHA: `19daa0ed41947a10ca327f5c575cff12dba6ac8354cb7433dbeac562f6205399`.
Полный audit можно воспроизвести на f3aad61: там есть runtime auditor и
исходные 447 файлов. После изменения fixture/lineage code текущий checkout
уже не совпадает с исходным манифестом; сохранённый pilot PASS не переписывать.

| Вариант | NDCG@10 | HR@10 | Best / epochs | First27 |
|---|---:|---:|---:|---:|
| baseline_dual | .0633 | .1162 | 27 /39 | .0620 |
| relative_phase | .0616 | .1124 | 11 /23 | N/A |
| absolute_phase | .0635 | .1174 | 54 /66 | .0620 |

Primary absolute−relative +.0019 (+3.0844%); absolute−baseline +.0002
(+.3160%); relative−baseline −.0017 (−2.6856%). Это exploratory pilot.
CPU100+100, targeted GPU17/532, smoke3×3 PASS. Baseline history и checkpoint
SHA точно совпали с историческим контролем. TEST0. Weights остались на cluster.

Артефакты: `evidence/job4374917/`. Transport helper первоначально отказался
из-за cap20MB для35.2MB absolute JSON; cap исправлен до100MB, файл сохранён
целиком. Это перенос evidence, не GPU retry. Подробности:
`runtime/preservation_transfer_review.json`.

## Confirmation

Frozen rule выполнено: 3 complete paired audited fits, absolute>=обоих
контролей. `runtime/confirmation_decision.json` содержит исходное решение;
execution binding пока PENDING_REVIEWED_WRAPPER_REVISION. Нужны 12 новых fits:
seeds2027–2030, на каждом baseline→relative→absolute. Настройки науки неизменны.

До пилотных метрик найден и сохранён реальный confirmation CPU FAIL:
100 tests,29 failing subcases,2 errors в12 fixture methods. Фикстуры оставляли
SEEDS2027–2030, создавая записи2026. Evidence:
`evidence/confirmation_cpu_scope_failure/`. Это0GPUsubmit и0scientificfits.

Gate agent исправляет ровно четыре frozen paths: test_failure_records.py,
test_protocol.py, config.py (только manifest locator), provenance.py (lineage).
Все остальные443 файла должны остаться побайтно прежними. Новый manifest:
source_manifest_confirmation.json. Старый plan/math/model/training/gate/policy
не изменять. Separate review и negative lineage tests обязательны перед freeze.

Lineage связывает original/new manifests и commits, точные4 changes,443
unchanged hashes, исходный CPU FAIL и независимый transition review.
Decision.source_hash/execution_commit остаются PILOT; новые
confirmation_source_hash/confirmation_execution_commit относятся к новой
версии. Runtime decision/lineage/review находятся вне frozen manifest;
их окончательные exact bytes переносятся на cluster после deploy candidate,
затем связываются с login/reservation. Не создавать циклический self-commit SHA.

Fit-audit agent адаптирует preserve_terminal.py, audit_saved.py,
append_registry.py к отдельному confirmation manifest и lineage. Сохранять
lineage, review, оба manifests и исходный CPU failure вместе с terminal evidence.

## Публикация

Реестр реально обновлён125→128. Исходный prefix58287 bytes сохранён:
SHA30ef3c9c58c90662367fe21de5fcc8d71315d6d475564ec0c6d9b1adb8053c63.
Новый CSV SHA39aaa1219caeda7f3a4552d7ff1178944b8c1912e0c2254ad90c222190508d58.
Package RESULTS и два main reports обновлены; независимый финальный reviewer
проверяет пилотную публикацию. Полный пилот разрешено merge/push в main.
Incomplete confirmation остаётся в ветке с partial report.

## Бюджет и следующий шаг

Использовано1 submit,3 scientific fits,6h requested GPU. Максимум3submit,
15startedfits,18h requested; confirmation8h, отдельный глобальный retry4h
только после доказанного pre-fit wrapper failure. Начатые fits не повторять.
Сейчас pilot terminal; checkout cluster можно обновить после exact diff review.

1. Закончить review pilot reports/CSV и narrow4-file correction.
2. Заморозить новый447-file confirmation manifest и candidate execution.
3. Сформировать lineage/review/decision с точными SHA; сохранить original pilot.
4. Развернуть candidate в единственном canonical checkout. Проверить CPU100/100
   и no-Git100/100 именно в confirmation stage, coverage и reservation ledger.
5. Отправить ровно один confirmation job на12fits/8h, продолжать до terminal.
6. Полный независимый audit, four-new и all-five aggregates отдельно,
   first27 только полные окна, TeX+paired primary SVG, registry140 при12/12.

После JobID polling не чаще600s. Очередь в сессии ждать максимум4h.
Не менять checkout под running/pending job. Не создавать clones/worktrees,
не менять environment/SSH, не использовать reset/clean/force-push.
Статью, Overleaf, draw.io не менять; сообщения людям не отправлять.
Следующие B/C/E/F — только roadmap, без реализации или GPU.
Не завершать всю задачу на pilot или SUBMITTED: разрешённая confirmation нужна.
