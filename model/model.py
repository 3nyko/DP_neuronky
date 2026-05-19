import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, cat
from base import BaseModel

#NUM_CLASSES = 2 # BENIGN, ATTACK
NUM_CLASSES = 6 # BENIGN, DOS, GAS, RPM, SPEED, STEERING_WHEEL
INPUT_DIM = 9 # ID + DATA_0 až DATA_7

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

class model_tab_ft_transformer(BaseModel):
    """
    FT-Transformer-style model: each scalar feature is projected by its own linear layer (feature tokenizer),
    optional column embeddings, Transformer encoder over the sequence, classify from a CLS token.
    Helps when interactions among CAN signals matter beyond fixed MLP geometry.
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.15):
        super().__init__()
        assert d_model % nhead == 0
        self.feature_tokenizers = nn.ModuleList([nn.Linear(1, d_model) for _ in range(INPUT_DIM)])
        self.col_embedding = nn.Parameter(torch.randn(1, INPUT_DIM, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_CLASSES),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        b = x.size(0)
        tokens = torch.stack([self.feature_tokenizers[i](x[:, i : i + 1]) for i in range(INPUT_DIM)], dim=1)
        tokens = tokens + self.col_embedding
        cls = self.cls_token.expand(b, -1, -1)
        h = cat([cls, tokens], dim=1)
        h = self.encoder(h)
        cls_out = h[:, 0]
        cls_out = self.norm_out(cls_out)
        return F.log_softmax(self.head(cls_out), dim=1)

class model_CNN_multiscale(BaseModel):
    """
    Parallel dilated 1D branches over the CAN frame (length INPUT_DIM). Different dilations cover
    multi-scale local patterns without collapsing length early; fusion + depthwise-separable style
    bottleneck keeps parameters reasonable.
    """

    def __init__(self, branch_ch=40, fused_ch=128, dropout=0.25):
        super().__init__()
        ch = branch_ch * 3
        self.branch_d1 = nn.Sequential(
            nn.Conv1d(1, branch_ch, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(),
        )
        self.branch_d2 = nn.Sequential(
            nn.Conv1d(1, branch_ch, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(),
        )
        self.branch_d4 = nn.Sequential(
            nn.Conv1d(1, branch_ch, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(branch_ch),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(ch, fused_ch, kernel_size=1),
            nn.BatchNorm1d(fused_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(fused_ch, fused_ch, kernel_size=3, padding=1, groups=fused_ch),
            nn.BatchNorm1d(fused_ch),
            nn.ReLU(),
        )
        self.gap_avg = nn.AdaptiveAvgPool1d(1)
        self.gap_max = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_ch * 2, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(96, NUM_CLASSES),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1).unsqueeze(1)
        y1 = self.branch_d1(x)
        y2 = self.branch_d2(x)
        y3 = self.branch_d4(x)
        y = cat([y1, y2, y3], dim=1)
        y = self.fuse(y)
        ya = self.gap_avg(y).squeeze(-1)
        ym = self.gap_max(y).squeeze(-1)
        y = cat([ya, ym], dim=1)
        return F.log_softmax(self.fc(y), dim=1)

class model_CNN_LSTM(BaseModel):
    def __init__(self, input_channels=1, window_size=2048, num_classes=None):
        '''
        1D CNN + LSTM. With CICIoV2024_DataLoader, batches are ``[B, INPUT_DIM]``; they are reshaped to
        ``[B, 1, INPUT_DIM]`` so Conv1d sees one channel and length INPUT_DIM (kernels sized for short sequences).

        For multichannel time series, pass ``input_channels=C`` and supply ``[B, C, T]`` or ``[B, C, 1, T]``.
        '''
        super().__init__()
        if num_classes is None:
            num_classes = NUM_CLASSES

        conv_channels_1 = 16
        self.conv1 = nn.Conv1d(input_channels, conv_channels_1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_channels_1)

        conv_channels_2 = 64
        self.conv2 = nn.Conv1d(conv_channels_1, conv_channels_2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_channels_2)

        conv_channels_3 = 128
        self.conv3 = nn.Conv1d(conv_channels_2, conv_channels_3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(conv_channels_3)
        out_channel = conv_channels_3

        self.pool = nn.MaxPool1d(kernel_size=2)

        hidden_dim = 64
        bidir = False
        self.lstm = nn.LSTM(input_size=out_channel, hidden_size=hidden_dim, batch_first=True, bidirectional=bidir)

        self.fc = nn.Linear(hidden_dim * (1 + int(bidir)), num_classes)

    def forward(self, x):
        # CICIoV default collate: [B, INPUT_DIM] → Conv1d would treat as unbatched [C,L] → wrong [1,B,L].
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x.view(x.size(0), -1).unsqueeze(1)
        elif x.dim() != 3:
            x = x.view(x.size(0), -1).unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.pool(x)

        x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)

        if self.lstm.bidirectional:
            x = torch.cat((h_n[0], h_n[1]), dim=1)
        else:
            x = h_n[-1]

        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# =====================================================
# =========       Autoencoder models          =========
# =====================================================

class model_autoencoder_shallow(BaseModel):
    """
    Simple symmetric autoencoder. Bottleneck forces compression;
    trained with MSE on BENIGN only, anomalies produce high reconstruction error.
    """

    def __init__(self, input_dim=INPUT_DIM, bottleneck=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return self.decoder(z)

class model_autoencoder_deep(BaseModel):
    """
    Deeper autoencoder with batch norm and dropout for better generalization.
    Narrower bottleneck forces more abstract representation of normal traffic.
    """

    def __init__(self, input_dim=INPUT_DIM, bottleneck=3, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return self.decoder(z)

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

class VectorResBlock(nn.Module):
    """Pre-norm residual block for fixed-size vectors (1D batch norm over features)."""

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


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
    
