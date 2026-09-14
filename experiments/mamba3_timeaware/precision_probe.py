"""CPU synthetic dataset regression: exact history time, unchanged baseline split."""

from pathlib import Path
import tempfile

import torch
from recbole.config import Config
from recbole.data.dataset import SequentialDataset
from recbole.model.abstract_recommender import SequentialRecommender

from .config import load_config
from .dataset import PreciseHistoryDataset


class HistoryProbe(SequentialRecommender):
    pass


def main():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / 'tiny'
        folder.mkdir()
        lines = ['user_id:token\titem_id:token\ttimestamp:float']
        for user in (1, 2):
            for event in range(7):
                lines.append(f'{user}\t{event + 1}\t{1649475963278 + event * 10 + user}')
        (folder / 'tiny.inter').write_text('\n'.join(lines) + '\n')
        settings = load_config()
        settings.update(dataset='tiny', data_path=directory, use_gpu=False,
                        user_inter_num_interval='[0,inf)', item_inter_num_interval='[0,inf)')
        config = Config(model=HistoryProbe, config_dict=settings)
        baseline = SequentialDataset(config).build()
        precise = PreciseHistoryDataset(config).build()
        for old, new in zip(baseline, precise):
            for field in ('user_id', 'item_id', 'item_id_list', 'item_length', 'timestamp'):
                assert torch.equal(old.inter_feat[field], new.inter_feat[field]), field
            assert new.inter_feat['timestamp_list'].dtype == torch.float64
            assert '_rt_exact_timestamp' not in new.inter_feat
        history = precise[0].inter_feat['timestamp_list']
        lengths = precise[0].inter_feat['item_length']
        assert ((history[:, 1] - history[:, 0])[lengths >= 2] == 10).all()
        print('PASS: synthetic item histories/targets/split unchanged; exact 10ms historical gaps preserved')


if __name__ == '__main__':
    main()
