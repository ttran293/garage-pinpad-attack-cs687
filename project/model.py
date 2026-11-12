import timm
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F

if __package__:
    from .enhanced_features import BlockScaler, HARMONIC_BLOCKS
else:
    from enhanced_features import BlockScaler, HARMONIC_BLOCKS


class KeypadCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        return self.layers(images)


class KeypadCoAtNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = timm.create_model(
            "coatnet_0_224",
            pretrained=False,
            in_chans=1,
            num_classes=10,
            img_size=64,
        )

    def forward(self, images):
        # Each press produces a 64-band mel-spectrogram. CoAtNet uses 64x64 images.
        images = F.interpolate(images, size=(64, 64), mode="bilinear", align_corners=False)
        return self.network(images)


class KeypadGRU(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.context = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.output = nn.Linear(128, 10)

    def forward(self, features):
        context, _ = self.context(self.encoder(features))
        return self.output(context)


def make_keypad_svm():
    return make_pipeline(BlockScaler(HARMONIC_BLOCKS), SVC(C=10, kernel="rbf", gamma="scale"))
