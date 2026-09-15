"""Preserve exact history milliseconds without changing RecBole sorting keys."""

import torch
from recbole.data.dataset import SequentialDataset
from recbole.utils import FeatureType, FeatureSource


class PreciseHistoryDataset(SequentialDataset):
    _precise = '_rt_exact_timestamp'

    def _dataframe_to_interaction(self, data):
        result = super()._dataframe_to_interaction(data)
        if self.time_field in data:
            self.set_field_property(self._precise, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
            result[self._precise] = torch.tensor(data[self.time_field].to_numpy(), dtype=torch.float64)
        return result

    def data_augmentation(self):
        # Upstream sort still uses the original float32 timestamp, unchanged.
        super().data_augmentation()
        precise_list = self._precise + self.config['LIST_SUFFIX']
        self.inter_feat[self.time_field + self.config['LIST_SUFFIX']] = self.inter_feat[precise_list]
        # Never expose the auxiliary exact target timestamp to the recommender.
        for field in (self._precise, precise_list):
            del self.inter_feat[field]
            for mapping in (self.field2type, self.field2source, self.field2seqlen):
                mapping.pop(field, None)
