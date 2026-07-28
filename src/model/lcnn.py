import torch
import torch.nn as nn
import torchaudio

class MFM(nn.Module):
    """Max-Feature-Map (MFM) activation function."""
    def __init__(self, dim=1):
        super(MFM, self).__init__()
        self.dim = dim

    def forward(self, x):
        channels = x.size(self.dim) // 2
        return torch.max(
            x.narrow(self.dim, 0, channels),
            x.narrow(self.dim, channels, channels)
        )

def createLFB(n_fft, sr, n_filters):
    """Linear Filter Bank (LFB) initialization tensor."""
    # return Tensor: [n_fft//2+1, n_filters]
    n_bins = n_fft // 2 + 1
    f = torch.linspace(0, sr / 2, n_bins).unsqueeze(1) # [n_bins, 1]
    arr = torch.linspace(0, sr / 2, n_filters + 2)     # [n_filters + 2]
    
    a = arr[:-2]
    b = arr[1:-1]
    c = arr[2:]
    up = (f - a) / (b - a)
    down = (c - f) / (c - b)
    
    answer = torch.clamp(torch.min(up, down), min=0.0)
    return answer

class LCNN(nn.Module):
    """LCNN for anti-spoofing task."""
    def __init__(self, n_fft=512, win_length=320, hop_length=160, sr=16000, dropout_rate=0.75):
        super(LCNN, self).__init__()
        
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.blackman_window,
            power=1.0
        )

        self.compress = nn.Linear(n_fft//2+1, 60, bias=False)
        self.compress.weight = nn.Parameter(createLFB(n_fft, sr, 60).T)
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 2),
            MFM(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 1, 1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, 3, 1, 1),
            MFM(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 96, 1, 1),
            MFM(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, 3, 1, 1),
            MFM(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 1, 1),
            MFM(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            MFM(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 1, 1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1),
            MFM(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(p=dropout_rate),
            nn.LazyLinear(out_features=160),
            MFM(),
            nn.Linear(80, 1)
        )
    
    def forward(self, audio, **kwargs):
        """Forward pass.

        Args:
            audio (Tensor): audio wave tensor of shape [B, Time].

        Returns:
            dict: Dictionary containing 'logits' tensor of shape [B].
        """
        x = self.stft(audio) # x.shape = [B, n_fft//2+1, Time]
        
        x = x.transpose(1, 2) # [B, Time, n_fft//2+1]
        x = self.compress(x)  # [B, Time, 60]
        x = torch.pow(x, 2)
        x = torch.log10(torch.clamp(x, min=1.2e-7)) # [B, Time, 60]
        
        x = x.transpose(1, 2) # [B, 60, Time]
        x = x.unsqueeze(dim=1) # [B, 1, 60, Time]
        
        x = self.features(x)
        x = self.classifier(x) # [B, 1]
        x = x.squeeze(1) # [B] NOT of probabilities
        return {'logits': x}
