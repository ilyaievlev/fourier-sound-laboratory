"""Small machine-ID classifier; all architecture parameters are exportable."""
import torch
from torch import nn


class PumpIDCNN(nn.Module):
    def __init__(self, num_ids=4, channels=(32, 64, 128, 192), input_offset=40., input_scale=40.):
        super().__init__()
        self.input_offset, self.input_scale = input_offset, input_scale
        layers, previous = [], 1
        for width in channels:
            layers += [nn.Conv2d(previous, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
                       nn.ReLU(), nn.MaxPool2d(2)]
            previous = width
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(previous, num_ids)

    def forward(self, values):
        values = (values + self.input_offset) / self.input_scale
        return self.head(self.encoder(values[:, None]).mean(dim=(2, 3)))


def from_config(config):
    if config['architecture'] != 'PumpIDCNN':
        raise ValueError('Unknown anomaly model')
    return PumpIDCNN(config['num_ids'], config['channels'], config['input_offset'], config['input_scale'])


def true_id_score(logits, true_id):
    """Mean negative log true-ID probability across crops, not -log(mean p)."""
    return -torch.log_softmax(logits.float(), dim=-1)[:, true_id].mean()
