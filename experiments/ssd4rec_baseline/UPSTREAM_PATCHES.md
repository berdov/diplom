# SSD4Rec upstream changes

Upstream snapshot:

- repository: `https://github.com/ZhangYifeng1995/SSD4Rec`
- commit: `bdbfe5193f3a6697bb6ee0699ab43386d80c6198`
- commit date: `2025-01-16T08:48:49Z`
- license: MIT License, copyright 2025 Zhang Yifeng

Files copied into `experiments/ssd4rec_baseline/upstream/`:

- `LICENSE`
- `README.md`
- `config.yaml`
- `environment.yaml`
- `main.py`
- `ssd4rec.py`
- `custom_utils.py`
- `custom_trainer.py`

Files intentionally not copied:

- `.git/`
- upstream `dataset/`
- `.gitignore`

Semantic patches to upstream files: none.

Mechanical normalization applied before commit:

- removed trailing whitespace;
- normalized final newline.

The project smoke wrapper applies two runtime compatibility shims without editing
the upstream snapshot:

- `np.float = float` for NumPy versions where `np.float` has been removed;
- `custom_utils.getLogger = logging.getLogger`, because upstream `custom_utils.py`
  calls `getLogger()` without importing it explicitly.
