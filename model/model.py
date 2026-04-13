import torch.nn as nn
import torch.nn.functional as F
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
