import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, cat
from base import BaseModel

#NUM_CLASSES = 2 # BENIGN, ATTACK
NUM_CLASSES = 6 # BENIGN, DOS, GAS, RPM, SPEED, STEERING_WHEEL
INPUT_DIM = 8 # DATA_0 až _7

# =====================================================
# =========        CICIoV2024 modely          =========
# =====================================================

class model_NN_1(BaseModel): 
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # pokud přijde tensor tvaru [B, 1, 1, N], rozvineme ho
        x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x
    
class model_CNN_1(BaseModel):
    def __init__(self):
        super().__init__()

        # 1D konvoluční část
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)

        # velikost po conv/pool části spočítáme z INPUT_DIM
        conv_output_dim = INPUT_DIM

        conv_output_dim = conv_output_dim // 2   # po prvním poolu
        conv_output_dim = conv_output_dim // 2   # po druhém poolu
        conv_output_dim = conv_output_dim // 2   # po třetím poolu

        self.fc1 = nn.Linear(128 * conv_output_dim, 256)
        self.fc2 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        # očekáváme vstup [B, 1, 1, N] nebo [B, N]
        if x.dim() == 4:
            x = x.view(x.size(0), -1)   # [B, N]

        # Conv1d chce vstup [B, C, L]
        x = x.unsqueeze(1)  # [B, 1, N]

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

class model_CNN_light(BaseModel):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool1d(1)   # zkrátí délku na 1
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        # [B, 1, 1, N] -> [B, N]
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        # [B, N] -> [B, 1, N]
        x = x.unsqueeze(1)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        x = self.gap(x)          # [B, 64, 1]
        x = x.squeeze(-1)        # [B, 64]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

class model_CNN_residual(BaseModel):
    
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.block1 = ResidualBlock1D(32, 64, kernel_size=5, dropout=0.15)
        self.pool1 = nn.MaxPool1d(2)

        self.block2 = ResidualBlock1D(64, 128, kernel_size=5, dropout=0.15)
        self.pool2 = nn.MaxPool1d(2)

        self.block3 = ResidualBlock1D(128, 128, kernel_size=3, dropout=0.15)
        self.pool3 = nn.MaxPool1d(2)

        self.block4 = ResidualBlock1D(128, 256, kernel_size=3, dropout=0.15)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        # očekává [B, 1, 1, N] nebo [B, N]
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        x = x.unsqueeze(1)  # [B, 1, N]

        x = self.stem(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.block4(x)

        x = self.gap(x)         # [B, 256, 1]
        x = x.squeeze(-1)       # [B, 256]

        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout_fc(x)

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

class model_NN_2(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(INPUT_DIM, 192)
        self.bn1 = nn.BatchNorm1d(192)

        self.fc2 = nn.Linear(192, 160)
        self.bn2 = nn.BatchNorm1d(160)

        self.fc3 = nn.Linear(160, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, 96)
        self.bn4 = nn.BatchNorm1d(96)

        self.fc5 = nn.Linear(96, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.fc6 = nn.Linear(64, NUM_CLASSES)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)

        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)

        return x

class model_NN_3(BaseModel):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(INPUT_DIM, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 224)
        self.bn2 = nn.BatchNorm1d(224)

        self.fc3 = nn.Linear(224, 160)
        self.bn3 = nn.BatchNorm1d(160)

        self.fc4 = nn.Linear(160, 96)
        self.bn4 = nn.BatchNorm1d(96)

        self.fc5 = nn.Linear(96, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.fc6 = nn.Linear(64, NUM_CLASSES)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.30)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout2(x)

        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)

        return x

class model_CNN_2(BaseModel):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 96, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(96)

        self.conv4 = nn.Conv1d(96, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.dropout_conv = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.3)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        # očekáváme [B, 1, 1, N] nebo [B, N]
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)

        # Conv1d chce [B, C, L]
        x = x.unsqueeze(1)  # [B, 1, N]

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)

        x = self.gap(x)          # [B, 128, 1]
        x = x.squeeze(-1)        # [B, 128]

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x

class model_CNN_3(BaseModel):
    def __init__(self):
        super().__init__()

        # blok 1
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        # blok 2
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)

        # blok 3
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(128)

        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.dropout_conv = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128, 192)
        self.bn_fc1 = nn.BatchNorm1d(192)

        self.fc2 = nn.Linear(192, 96)
        self.bn_fc2 = nn.BatchNorm1d(96)

        self.fc3 = nn.Linear(96, NUM_CLASSES)

    def forward(self, x):
        # [B, 1, 1, N] nebo [B, N] -> [B, N]
        x = x.view(x.size(0), -1)

        # Conv1d chce [B, C, L]
        x = x.unsqueeze(1)  # [B, 1, N]

        # blok 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # blok 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # blok 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout_conv(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)

        # globální pooling místo velkého flattenu
        x = self.gap(x)      # [B, 128, 1]
        x = x.squeeze(-1)    # [B, 128]

        # klasifikační hlava
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x

class model_CNN_SE_Res(BaseModel):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.block1 = ResidualSEBlock1D(32, 48, kernel_size=5, dropout=0.10)
        self.pool1 = nn.MaxPool1d(2)

        self.block2 = ResidualSEBlock1D(48, 80, kernel_size=5, dropout=0.15)
        self.pool2 = nn.MaxPool1d(2)

        self.block3 = ResidualSEBlock1D(80, 128, kernel_size=3, dropout=0.20)
        self.pool3 = nn.MaxPool1d(2)

        self.block4 = ResidualSEBlock1D(128, 128, kernel_size=3, dropout=0.20)

        self.gap_avg = nn.AdaptiveAvgPool1d(1)
        self.gap_max = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(128 * 2, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.35)

        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        # [B, 1, 1, N] nebo [B, N]
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)  # [B, 1, N]

        x = self.stem(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.block4(x)

        x_avg = self.gap_avg(x).squeeze(-1)   # [B, 128]
        x_max = self.gap_max(x).squeeze(-1)   # [B, 128]
        x = cat([x_avg, x_max], dim=1)  # [B, 256]

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x

# =====================================================
# =========           MNIST model             =========
# =====================================================
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# =====================================================
# =========           Functions               =========
# =====================================================

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out)

        return out
    
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: [B, C, L]
        s = x.mean(dim=2)                 # [B, C]
        s = F.relu(self.fc1(s))
        s = sigmoid(self.fc2(s))    # [B, C]
        s = s.unsqueeze(2)                # [B, C, 1]
        return x * s

class ResidualSEBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.15):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock1D(out_channels, reduction=8)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = out + identity
        out = F.relu(out)
        return out
    
