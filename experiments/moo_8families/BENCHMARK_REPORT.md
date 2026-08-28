# MOO 8 Families Benchmark Report

Статус: scaffold and runnable scripts prepared; cluster smoke/sanity results are not recorded yet in this file.

## Safety

- Protocol B fingerprint должен совпасть с `23951/7111/1134420/1086518/23951/23951`.
- Identity hash должен совпасть с `954d8abff424b5a57daa74f361ab0f8309cf93121fcc12ef10569d2df11144c7`.
- Test split закрыт. Все новые строки benchmark должны иметь `test_evaluated=false`.
- PCGrad учитывается как historical validation-only run `pcgrad_001`; автоматический rerun запрещен.

## Next Gate

1. Запустить 7 smoke jobs на `type_e`.
2. Проверить, что все smoke jobs завершились и не трогали test.
3. Только затем запускать 5-epoch sanity jobs.
4. После sanity выполнить:

```bash
python -m experiments.moo_8families.build_results --write-report
```

Итоговые таблицы появятся здесь после `build_results.py`.
