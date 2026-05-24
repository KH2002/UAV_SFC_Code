# -*- coding: utf-8 -*-
"""
Transformer 编码器模块
- UAV Transformer
- Request Transformer
- 位置编码
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    用于为Transformer提供序列位置信息
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class UAVTransformer(nn.Module):
    """
    UAV Transformer编码器
    
    注意：UAV序列的顺序是任意的（按ID排列），没有语义信息，
    因此不使用位置编码。
    
    输入: [batch, NUM_UAVS, 6] - UAV特征向量 (energy, cpu, pos_x, pos_y, is_busy, dist_to_base)
    输出: [batch, NUM_UAVS, hidden_dim] - UAV嵌入表示
    """
    def __init__(self, 
                 input_dim: int = 6, 
                 hidden_dim: int = 64, 
                 num_heads: int = 4, 
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # 注意：不使用位置编码，因为UAV的ID顺序是任意的
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # LayerNorm用于稳定训练
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, uav_states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            uav_states: [batch_size, num_uavs, input_dim]
            mask: [batch_size, num_uavs] - 可选的padding mask
        
        Returns:
            uav_embeds: [batch_size, num_uavs, hidden_dim]
        """
        # 输入嵌入
        x = self.embedding(uav_states)  # [batch, num_uavs, hidden_dim]
        
        # 注意：不添加位置编码，因为UAV的ID顺序是任意的
        
        # Transformer编码
        if mask is not None:
            # mask: True表示需要mask的位置
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # LayerNorm
        x = self.layer_norm(x)
        
        return x


class RequestTransformer(nn.Module):
    """
    Request Transformer编码器
    
    注意：请求序列的顺序是任意的，没有语义信息，因此不使用位置编码。
    
    输入: [batch, MAX_PENDING, 5] - 请求特征向量 (num_vnfs, workload, comm, loc1, loc2)
    输出: [batch, MAX_PENDING, hidden_dim] - 请求嵌入表示
    """
    def __init__(self, 
                 input_dim: int = 5, 
                 hidden_dim: int = 64, 
                 num_heads: int = 4, 
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # 注意：不使用位置编码，因为请求顺序是任意的
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, request_states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            request_states: [batch_size, max_pending, input_dim]
            mask: [batch_size, max_pending] - 可选的padding mask
        
        Returns:
            request_embeds: [batch_size, max_pending, hidden_dim]
        """
        # 输入嵌入
        x = self.embedding(request_states)  # [batch, max_pending, hidden_dim]
        
        # 注意：不添加位置编码
        
        # Transformer编码
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # LayerNorm
        x = self.layer_norm(x)
        
        return x


class CrossAttention(nn.Module):
    """
    请求-UAV交叉注意力模块
    
    用于计算请求与UAV之间的兼容性分数
    """
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 多头注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                query: torch.Tensor, 
                key_value: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len_q, hidden_dim] - 请求嵌入
            key_value: [batch, seq_len_kv, hidden_dim] - UAV嵌入
            attn_mask: 可选的注意力mask
        
        Returns:
            output: [batch, seq_len_q, hidden_dim]
            attn_weights: [batch, seq_len_q, seq_len_kv]
        """
        # 交叉注意力
        attn_output, attn_weights = self.cross_attn(
            query, key_value, key_value,
            key_padding_mask=attn_mask
        )
        
        # 残差连接 + LayerNorm
        x = self.norm1(query + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights
