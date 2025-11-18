import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, num_layers, seq_length, pre_len):
        """
        参数说明：
        - num_nodes: 节点数 (N)
        - in_dim: 每个节点的输入特征数 (例如：1)
        - hidden_dim: LSTM 隐藏状态维度
        - num_layers: LSTM 层数
        - seq_length: 输入历史序列长度 (T)
        - pre_len: 预测未来时间步数
        """
        super(LSTM, self).__init__()
        self.num_nodes = num_nodes      # 节点数 N
        self.in_dim = in_dim            # 每个节点输入特征数 C
        self.seq_length = seq_length    # 历史时间步长 T
        self.pre_len = pre_len          # 预测步数

        # 这里将每个时间步的输入融合为一个向量，其长度为 N * in_dim
        lstm_input_size = num_nodes * in_dim

        # 定义 LSTM 层，输入尺寸为 (N*in_dim)，输出隐藏状态为 hidden_dim
        # 设置 batch_first=True 表示输入形状为 (B, T, lstm_input_size)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)

        # 定义全连接层：
        # 将 LSTM 最后时间步的输出映射到 (N * pre_len) 维，再重塑为 (B, N, pre_len)
        self.fc = nn.Linear(hidden_dim, num_nodes * pre_len)

    def forward(self, x):
        """
        输入 x 的形状: (B, in_dim, N, seq_length)
          B: batch size
          in_dim: 每个节点的输入特征数 (C)
          N: 节点数
          seq_length: 历史时间步长 (T)

        修改后的步骤说明：
        1. 先将输入调整为 (B, seq_length, N, in_dim)
           - 通过 permute 将时间轴移到第二个维度
        2. 将节点与 in_dim 维度融合，得到形状 (B, seq_length, N * in_dim)
        3. 将数据送入 LSTM，输出形状为 (B, seq_length, hidden_dim)
        4. 取最后一个时间步的输出，形状 (B, hidden_dim)
        5. 通过全连接层映射到 (B, N * pre_len)
        6. 重塑回 (B, N, pre_len)
        """
        x = x.permute(0, 2 ,1 ,3)
        B, C, N, T = x.shape  # 原始 x: (B, in_dim, N, seq_length)
        # 1. 将 x 调整为 (B, seq_length, N, in_dim)
        x = x.permute(0, 3, 2, 1)  # 现在 x 的形状为 (B, T, N, C)
        
        # 2. 将节点维度和特征维度融合为一个向量
        x = x.reshape(B, T, N * C)  # 形状为 (B, T, N*in_dim)
        
        # 3. 将数据送入 LSTM，输出 shape: (B, T, hidden_dim)
        lstm_out, _ = self.lstm(x)
        
        # 4. 取最后时间步输出，形状 (B, hidden_dim)
        last_hidden = lstm_out[:, -1, :]
        
        # 5. 全连接层映射到 (B, N * pre_len)
        pred = self.fc(last_hidden)
        
        # 6. 重塑回 (B, N, pre_len)
        pred = pred.reshape(B, N, self.pre_len)
        return pred
