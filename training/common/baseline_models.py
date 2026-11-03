"""Small waveform and log-mel CNNs with global average pooling."""
import librosa
import torch
from torch import nn


class WaveCNN(nn.Module):
    def __init__(self, classes=50):
        super().__init__()
        layers, previous = [], 1
        for channels, kernel, stride, pool in [(32, 80, 4, 4), (64, 9, 1, 4),
                                             (128, 9, 1, 4), (192, 9, 1, 4), (256, 3, 1, 2)]:
            layers.extend([nn.Conv1d(previous, channels, kernel, stride, kernel // 2, bias=False),
                           nn.BatchNorm1d(channels), nn.ReLU(), nn.MaxPool1d(pool)])
            previous = channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(previous, classes)

    def forward(self, wave):
        return self.head(self.encoder(wave[:, None]).mean(dim=-1))


class MelCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('window', torch.hann_window(config['n_fft']))
        self.register_buffer('mel_filter', torch.from_numpy(librosa.filters.mel(
            sr=config['sample_rate'], n_fft=config['n_fft'], n_mels=config['mel_bins'])))
        layers, previous = [], 1
        for channels in [32, 64, 128, 256]:
            layers.extend([nn.Conv2d(previous, channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2)])
            previous = channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(previous, config['num_classes'])

    def forward(self, wave):
        with torch.autocast(device_type=wave.device.type, enabled=False):
            power = torch.stft(wave.float(), n_fft=self.config['n_fft'],
                               hop_length=self.config['hop_length'], window=self.window,
                               return_complex=True).abs().square()
            mel = 10 * torch.log10(torch.matmul(self.mel_filter, power).clamp_min(1e-10))
            mel = (mel - mel.mean((1, 2), keepdim=True)) / mel.std((1, 2), keepdim=True).clamp_min(1e-5)
            if self.training:
                for axis, key in [(1, 'specaugment_frequency_width'), (2, 'specaugment_time_width')]:
                    for sample in mel:
                        width = int(torch.randint(self.config[key] + 1, ()).item())
                        start = int(torch.randint(mel.shape[axis] - width + 1, ()).item())
                        if axis == 1:
                            sample[start:start + width, :] = 0
                        else:
                            sample[:, start:start + width] = 0
        return self.head(self.encoder(mel[:, None]).mean(dim=(2, 3)))


def make_model(name, config):
    if name == 'wave_cnn':
        return WaveCNN(config['num_classes'])
    if name == 'mel_cnn':
        return MelCNN(config)
    raise ValueError(name)
