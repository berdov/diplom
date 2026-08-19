# Изменения относительно upstream SSD4Rec

Снимок upstream:

- репозиторий: `https://github.com/ZhangYifeng1995/SSD4Rec`;
- commit: `bdbfe5193f3a6697bb6ee0699ab43386d80c6198`;
- дата commit: `2025-01-16T08:48:49Z`;
- лицензия: MIT License, copyright 2025 Zhang Yifeng.

В `experiments/ssd4rec_baseline/upstream/` скопированы:

- `LICENSE`;
- `README.md`;
- `config.yaml`;
- `environment.yaml`;
- `main.py`;
- `ssd4rec.py`;
- `custom_utils.py`;
- `custom_trainer.py`.

Намеренно не копировались:

- `.git/`;
- upstream `dataset/`;
- `.gitignore`.

Математические или архитектурные патчи upstream-файлов: отсутствуют.

Механическая нормализация перед commit:

- удалены trailing whitespace;
- нормализован финальный newline.

Проектные wrappers `smoke_test.py` и `sanity_train.py` применяют два runtime
compatibility shim без редактирования upstream snapshot:

- `np.float = float` для NumPy-версий, где `np.float` уже удален;
- `custom_utils.getLogger = logging.getLogger`, потому что upstream
  `custom_utils.py` вызывает `getLogger()` без явного import.

Эти shims не меняют архитектуру, loss, masking, variable-length batching,
bidirectional SSD или evaluation semantics original SSD4Rec.
