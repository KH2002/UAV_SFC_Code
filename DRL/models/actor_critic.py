# -*- coding: utf-8 -*-
"""
Actor-Critic 网络模块
- Actor: 策略网络（请求选择 + UAV分配）
- Critic: 价值网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .transformers import UAVTransformer, RequestTransformer, CrossAttention


class Actor(nn.Module):
    """
    Actor网络：策略网络
    
    输出：
    1. 跳过概率 [batch, 1] - 是否跳过当前步进入下一时隙
    2. 请求选择概率分布 [batch, max_pending]
    3. UAV分配概率分布 [batch, num_uavs]（为选中的请求分配两个VNF的UAV）
    """
    def __init__(self, 
                 hidden_dim: int = 64,
                 num_uavs: int = 200,
                 max_pending: int = 20,
                 num_vnfs_per_request: int = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_uavs = num_uavs
        self.max_pending = max_pending
        self.num_vnfs_per_request = num_vnfs_per_request
        
        # 请求选择头
        self.request_selector = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # UAV选择头（为每个VNF选择UAV）
        # 输入：请求嵌入 + UAV嵌入（拼接）
        self.uav_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def _compute_request_logits(self, request_embeds: torch.Tensor, 
                                 masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """计算请求选择的logits"""
        request_logits = self.request_selector(request_embeds).squeeze(-1)
        if masks is not None and 'request' in masks:
            request_logits = request_logits.masked_fill(masks['request'] == 0, -1e9)
        return request_logits

    def _get_uav_mask_for_request(self,
                                  request_idx: torch.Tensor,
                                  masks: Optional[Dict[str, torch.Tensor]] = None,
                                  for_vnf: int = 1) -> Optional[torch.Tensor]:
        """
        获取与已选 request_idx 对应的 UAV 掩码。

        优先使用 request-aware 的 VNF专属掩码，若不存在则回退到并集 `uav_by_request`，
        最后回退到全局 `uav`。
        """
        if masks is None:
            return None

        batch_idx = torch.arange(request_idx.size(0), device=request_idx.device)

        if for_vnf == 1 and 'uav_by_request_vnf1' in masks:
            return masks['uav_by_request_vnf1'][batch_idx, request_idx]

        if for_vnf == 2 and 'uav_by_request_vnf2' in masks:
            return masks['uav_by_request_vnf2'][batch_idx, request_idx]

        if 'uav_by_request' in masks:
            return masks['uav_by_request'][batch_idx, request_idx]

        if 'uav' in masks:
            return masks['uav']

        return None
    
    def _compute_uav_logits_for_request(self, uav_embeds: torch.Tensor,
                                        request_embeds: torch.Tensor,
                                        request_idx: torch.Tensor,
                                        masks: Optional[Dict[str, torch.Tensor]] = None,
                                        for_vnf: int = 1) -> torch.Tensor:
        """
        为指定请求计算UAV选择的原始logits
        
        Args:
            uav_embeds: [batch, num_uavs, hidden_dim]
            request_embeds: [batch, max_pending, hidden_dim]
            request_idx: [batch] - 选中的请求索引
            masks: 动作掩码
        
        Returns:
            uav_logits: [batch, num_uavs]
        """
        batch_size = uav_embeds.size(0)
        selected_uav_mask = self._get_uav_mask_for_request(request_idx, masks, for_vnf=for_vnf)
        
        # 获取选中请求的嵌入
        selected_request_embeds = torch.stack([
            request_embeds[b, request_idx[b], :]
            for b in range(batch_size)
        ])
        
        # 计算UAV选择分数
        uav_logits_list = []
        for b in range(batch_size):
            req_emb = selected_request_embeds[b]
            uav_emb = uav_embeds[b]
            num_uavs_current = uav_emb.size(0)
            
            # 扩展请求嵌入
            req_emb_expanded = req_emb.unsqueeze(0).expand(num_uavs_current, -1)
            combined = torch.cat([req_emb_expanded, uav_emb], dim=-1)
            
            logits = self.uav_selector(combined).squeeze(-1)
            
            # 应用mask
            if selected_uav_mask is not None:
                logits = logits.masked_fill(selected_uav_mask[b] == 0, -1e9)
            
            uav_logits_list.append(logits)
        
        return torch.stack(uav_logits_list)
    
    def forward(self, 
                uav_embeds: torch.Tensor,
                request_embeds: torch.Tensor,
                masks: Optional[Dict[str, torch.Tensor]] = None,
                temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 仅用于获取各动作的边际分布（不推荐用于采样）
        
        注意：此方法使用 argmax 来近似条件分布，仅用于快速推断。
        训练时应该使用 get_action() 方法。
        """
        batch_size = uav_embeds.size(0)
        
        # ========== 请求选择 ==========
        request_logits = self._compute_request_logits(request_embeds, masks)
        request_probs = F.softmax(request_logits / temperature, dim=-1)
        
        # ========== UAV选择（基于 argmax request）==========
        # 注意：这里使用 argmax 只是为了快速推断，不是真实采样
        selected_request_idx = torch.argmax(request_probs, dim=-1)
        
        # 计算 VNF1 的 logits
        uav_logits_vnf1 = self._compute_uav_logits_for_request(
            uav_embeds, request_embeds, selected_request_idx, masks, for_vnf=1
        )
        uav_probs_vnf1 = F.softmax(uav_logits_vnf1 / temperature, dim=-1)
        
        # 基于 argmax VNF1 计算 VNF2
        selected_uav_for_vnf1 = torch.argmax(uav_probs_vnf1, dim=-1)
        uav_logits_vnf2 = self._compute_uav_logits_for_request(
            uav_embeds, request_embeds, selected_request_idx, masks, for_vnf=2
        )
        # 屏蔽 VNF1 已选的 UAV
        for b in range(batch_size):
            uav_logits_vnf2[b, selected_uav_for_vnf1[b]] = -1e9
        uav_probs_vnf2 = F.softmax(uav_logits_vnf2 / temperature, dim=-1)
        
        return {
            'request_probs': request_probs,
            'uav_probs_vnf1': uav_probs_vnf1,
            'uav_probs_vnf2': uav_probs_vnf2,
            'request_logits': request_logits,
            'uav_logits_vnf1': uav_logits_vnf1,
            'uav_logits_vnf2': uav_logits_vnf2,
            'selected_request_idx': selected_request_idx,
        }

    
    def get_action(self, 
                   uav_embeds: torch.Tensor,
                   request_embeds: torch.Tensor,
                   masks: Optional[Dict[str, torch.Tensor]] = None,
                   deterministic: bool = False,
                   epsilon: float = 0.0) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        采样或选择动作 - 使用正确的条件分布，支持 ε-贪心探索
        
        采样顺序：
        1. 采样 request_idx（支持 ε-贪心从有效请求中随机选）
        2. 基于采样的 request_idx，计算 VNF1 的分布并采样 uav_for_vnf1
        3. 基于实际采样的 uav_for_vnf1，计算 VNF2 的分布并采样 uav_for_vnf2
        
        这确保了 log_prob 计算与条件分布一致。
        
        Args:
            uav_embeds: UAV嵌入 [batch, num_uavs, hidden_dim]
            request_embeds: 请求嵌入 [batch, max_pending, hidden_dim]
            masks: 动作掩码 {
                'request': [batch, max_pending],
                'uav': [batch, num_uavs],
                'uav_by_request_vnf1': [batch, max_pending, num_uavs] (可选),
                'uav_by_request_vnf2': [batch, max_pending, num_uavs] (可选),
                'uav_by_request': [batch, max_pending, num_uavs] (兼容回退)
            }
            deterministic: 是否确定性选择（贪心）
            epsilon: ε-贪心探索概率，0 表示不探索，1 表示完全随机
        
        Returns:
            action: {
                'request_idx': [batch],
                'uav_for_vnf1': [batch],
                'uav_for_vnf2': [batch]
            }
            info: 包含log概率等信息的字典
        """
        batch_size = uav_embeds.size(0)
        device = uav_embeds.device
        
        # ========== Step 1: 请求选择 ==========
        request_logits = self._compute_request_logits(request_embeds, masks)
        request_probs = F.softmax(request_logits, dim=-1)
        
        if deterministic:
            request_idx = torch.argmax(request_probs, dim=-1)
        elif epsilon > 0 and torch.rand(1).item() < epsilon:
            # ε-贪心：从有效请求中均匀随机选择
            if masks is not None and 'request' in masks:
                request_mask = masks['request']  # [batch, max_pending]
                # 对每个 batch 单独采样
                request_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
                for b in range(batch_size):
                    valid_indices = torch.where(request_mask[b])[0]
                    if len(valid_indices) > 0:
                        request_idx[b] = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                    else:
                        request_idx[b] = 0  # 如果没有有效请求，选第一个（会被环境拒绝）
            else:
                # 没有 mask 时，从所有请求中随机
                request_idx = torch.randint(0, self.max_pending, (batch_size,), device=device)
        else:
            request_idx = torch.multinomial(request_probs, 1).squeeze(-1)
        
        # 计算 request 的 log_prob（即使随机采样也计算，用于 PPO 更新）
        log_prob_request = torch.log(request_probs.gather(1, request_idx.unsqueeze(1)).squeeze(1) + 1e-10)
        
        # ========== Step 2: VNF1 选择（基于实际采样的 request_idx）==========
        uav_logits_vnf1 = self._compute_uav_logits_for_request(
            uav_embeds, request_embeds, request_idx, masks, for_vnf=1
        )
        uav_probs_vnf1 = F.softmax(uav_logits_vnf1, dim=-1)
        selected_uav_mask = self._get_uav_mask_for_request(request_idx, masks, for_vnf=1)
        
        if deterministic:
            uav_for_vnf1 = torch.argmax(uav_probs_vnf1, dim=-1)
        elif epsilon > 0 and torch.rand(1).item() < epsilon:
            # ε-贪心：从有效 UAV 中均匀随机选择
            if selected_uav_mask is not None:
                uav_mask = selected_uav_mask  # [batch, num_uavs]
                uav_for_vnf1 = torch.zeros(batch_size, dtype=torch.long, device=device)
                for b in range(batch_size):
                    valid_indices = torch.where(uav_mask[b])[0]
                    if len(valid_indices) > 0:
                        uav_for_vnf1[b] = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                    else:
                        uav_for_vnf1[b] = 0
            else:
                uav_for_vnf1 = torch.randint(0, self.num_uavs, (batch_size,), device=device)
        else:
            uav_for_vnf1 = torch.multinomial(uav_probs_vnf1, 1).squeeze(-1)
        
        # 计算 VNF1 的 log_prob
        log_prob_uav1 = torch.log(uav_probs_vnf1.gather(1, uav_for_vnf1.unsqueeze(1)).squeeze(1) + 1e-10)
        
        # ========== Step 3: VNF2 选择（基于实际采样的 uav_for_vnf1）==========
        # 重新计算 VNF2 的 logits，并屏蔽实际采样的 uav_for_vnf1
        uav_logits_vnf2 = self._compute_uav_logits_for_request(
            uav_embeds, request_embeds, request_idx, masks, for_vnf=2
        )
        selected_uav_mask_vnf2 = self._get_uav_mask_for_request(request_idx, masks, for_vnf=2)
        
        # 屏蔽 VNF1 实际采样的 UAV
        for b in range(batch_size):
            uav_logits_vnf2[b, uav_for_vnf1[b]] = -1e9
        
        uav_probs_vnf2 = F.softmax(uav_logits_vnf2, dim=-1)
        
        if deterministic:
            uav_for_vnf2 = torch.argmax(uav_probs_vnf2, dim=-1)
        elif epsilon > 0 and torch.rand(1).item() < epsilon:
            # ε-贪心：从有效 UAV 中（排除 VNF1 已选的）均匀随机选择
            if selected_uav_mask_vnf2 is not None:
                uav_mask = selected_uav_mask_vnf2.clone()  # [batch, num_uavs]
                # 排除 VNF1 已选的 UAV
                for b in range(batch_size):
                    uav_mask[b, uav_for_vnf1[b]] = False
                
                uav_for_vnf2 = torch.zeros(batch_size, dtype=torch.long, device=device)
                for b in range(batch_size):
                    valid_indices = torch.where(uav_mask[b])[0]
                    if len(valid_indices) > 0:
                        uav_for_vnf2[b] = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                    else:
                        # 如果没有其他 UAV，只能选 VNF1 的 UAV（会被环境拒绝）
                        uav_for_vnf2[b] = (uav_for_vnf1[b] + 1) % self.num_uavs
            else:
                uav_for_vnf2 = torch.randint(0, self.num_uavs, (batch_size,), device=device)
                # 确保与 VNF1 不同
                for b in range(batch_size):
                    if uav_for_vnf2[b] == uav_for_vnf1[b]:
                        uav_for_vnf2[b] = (uav_for_vnf2[b] + 1) % self.num_uavs
        else:
            uav_for_vnf2 = torch.multinomial(uav_probs_vnf2, 1).squeeze(-1)
        
        # 计算 VNF2 的 log_prob
        log_prob_uav2 = torch.log(uav_probs_vnf2.gather(1, uav_for_vnf2.unsqueeze(1)).squeeze(1) + 1e-10)
        
        # 总 log_prob
        log_prob = log_prob_request + log_prob_uav1 + log_prob_uav2
        
        action = {
            'request_idx': request_idx,
            'uav_for_vnf1': uav_for_vnf1,
            'uav_for_vnf2': uav_for_vnf2
        }
        
        # 计算熵
        entropy_request = -(request_probs * torch.log(request_probs + 1e-10)).sum(dim=-1)
        entropy_uav1 = -(uav_probs_vnf1 * torch.log(uav_probs_vnf1 + 1e-10)).sum(dim=-1)
        entropy_uav2 = -(uav_probs_vnf2 * torch.log(uav_probs_vnf2 + 1e-10)).sum(dim=-1)
        entropy = entropy_request + entropy_uav1 + entropy_uav2
        
        info = {
            'log_prob': log_prob,
            'log_prob_request': log_prob_request,
            'log_prob_uav1': log_prob_uav1,
            'log_prob_uav2': log_prob_uav2,
            'entropy': entropy,
        }
        
        return action, info
    
    def _compute_entropy(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算策略的熵（用于鼓励探索）"""
        entropy_request = -(output['request_probs'] * torch.log(output['request_probs'] + 1e-10)).sum(dim=-1)
        entropy_uav1 = -(output['uav_probs_vnf1'] * torch.log(output['uav_probs_vnf1'] + 1e-10)).sum(dim=-1)
        entropy_uav2 = -(output['uav_probs_vnf2'] * torch.log(output['uav_probs_vnf2'] + 1e-10)).sum(dim=-1)
        
        return entropy_request + entropy_uav1 + entropy_uav2
    
    def evaluate_actions(self,
                         uav_embeds: torch.Tensor,
                         request_embeds: torch.Tensor,
                         actions: Dict[str, torch.Tensor],
                         masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        评估给定动作的log概率和熵（用于PPO更新）
        
        注意：此方法必须与 get_action 使用完全相同的条件分布计算逻辑，
        以确保重要性比率 ratio = exp(new_log_prob - old_log_prob) 是正确的。
        
        Args:
            uav_embeds: UAV嵌入 [batch, num_uavs, hidden_dim]
            request_embeds: 请求嵌入 [batch, max_pending, hidden_dim]
            actions: {
                'request_idx': [batch],
                'uav_for_vnf1': [batch],
                'uav_for_vnf2': [batch]
            }
            masks: 动作掩码
        
        Returns:
            {
                'log_prob': [batch],
                'entropy': [batch]
            }
        """
        batch_size = uav_embeds.size(0)
        
        # ========== Step 1: 请求选择的概率 ==========
        request_logits = self._compute_request_logits(request_embeds, masks)
        request_probs = F.softmax(request_logits, dim=-1)
        
        request_idx = actions['request_idx']
        log_prob_request = torch.log(request_probs.gather(1, request_idx.unsqueeze(1)).squeeze(1) + 1e-10)
        
        # ========== Step 2: VNF1 选择的概率（基于 actions['request_idx']）==========
        uav_logits_vnf1 = self._compute_uav_logits_for_request(
            uav_embeds, request_embeds, request_idx, masks, for_vnf=1
        )
        uav_probs_vnf1 = F.softmax(uav_logits_vnf1, dim=-1)
        
        uav_for_vnf1 = actions['uav_for_vnf1']
        log_prob_uav1 = torch.log(uav_probs_vnf1.gather(1, uav_for_vnf1.unsqueeze(1)).squeeze(1) + 1e-10)
        
        # ========== Step 3: VNF2 选择的概率（基于 actions['uav_for_vnf1']）==========
        # 重新计算 VNF2 的 logits，并屏蔽 actions['uav_for_vnf1'] 指定的 UAV
        uav_logits_vnf2 = self._compute_uav_logits_for_request(
            uav_embeds, request_embeds, request_idx, masks, for_vnf=2
        )
        
        # 屏蔽 VNF1 实际选择的 UAV
        for b in range(batch_size):
            uav_logits_vnf2[b, uav_for_vnf1[b]] = -1e9
        
        uav_probs_vnf2 = F.softmax(uav_logits_vnf2, dim=-1)
        
        uav_for_vnf2 = actions['uav_for_vnf2']
        log_prob_uav2 = torch.log(uav_probs_vnf2.gather(1, uav_for_vnf2.unsqueeze(1)).squeeze(1) + 1e-10)
        
        # 总 log_prob
        log_prob = log_prob_request + log_prob_uav1 + log_prob_uav2
        
        # 计算熵（用于探索奖励）
        entropy_request = -(request_probs * torch.log(request_probs + 1e-10)).sum(dim=-1)
        entropy_uav1 = -(uav_probs_vnf1 * torch.log(uav_probs_vnf1 + 1e-10)).sum(dim=-1)
        entropy_uav2 = -(uav_probs_vnf2 * torch.log(uav_probs_vnf2 + 1e-10)).sum(dim=-1)
        entropy = entropy_request + entropy_uav1 + entropy_uav2
        
        return {
            'log_prob': log_prob,
            'entropy': entropy
        }


class Critic(nn.Module):
    """
    Critic网络：价值网络
    
    输入：全局特征 + UAV嵌入均值 + 请求嵌入均值
    输出：状态价值 V(s)
    """
    def __init__(self, 
                 hidden_dim: int = 64,
                 global_dim: int = 4,
                 num_layers: int = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.global_dim = global_dim
        
        # 输入维度：global_dim + hidden_dim (UAV均值) + hidden_dim (Request均值)
        input_dim = global_dim + hidden_dim * 2
        
        # 构建网络
        layers = []
        
        # 第一层
        layers.append(nn.Linear(input_dim, 128))
        layers.append(nn.ReLU())
        
        # 中间层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(128, 64))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(64, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self,
                uav_embeds: torch.Tensor,
                request_embeds: torch.Tensor,
                global_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uav_embeds: [batch, num_uavs, hidden_dim]
            request_embeds: [batch, max_pending, hidden_dim]
            global_features: [batch, global_dim]
        
        Returns:
            value: [batch, 1]
        """
        # 计算UAV和Request嵌入的均值
        uav_mean = uav_embeds.mean(dim=1)  # [batch, hidden_dim]
        request_mean = request_embeds.mean(dim=1)  # [batch, hidden_dim]
        
        # 拼接特征
        combined = torch.cat([global_features, uav_mean, request_mean], dim=-1)
        
        # 前向传播
        value = self.network(combined)
        
        return value



class PolicyNetwork(nn.Module):
    """
    完整策略网络：整合Encoder、Actor、Critic
    
    架构：
    1. UAV Transformer编码器
    2. Request Transformer编码器
    3. 可选的交叉注意力
    4. Actor网络（策略）
    5. Critic网络（价值）
    """
    def __init__(self,
                 uav_input_dim: int = 6,
                 request_input_dim: int = 5,
                 global_dim: int = 4,
                 hidden_dim: int = 64,
                 num_uavs: int = 200,
                 max_pending: int = 20,
                 num_heads: int = 4,
                 num_encoder_layers: int = 2,
                 use_cross_attn: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_uavs = num_uavs
        self.max_pending = max_pending
        self.use_cross_attn = use_cross_attn
        
        # UAV编码器
        self.uav_encoder = UAVTransformer(
            input_dim=uav_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # Request编码器
        self.request_encoder = RequestTransformer(
            input_dim=request_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # 交叉注意力（可选）
        if use_cross_attn:
            self.cross_attn = CrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Actor和Critic
        self.actor = Actor(
            hidden_dim=hidden_dim,
            num_uavs=num_uavs,
            max_pending=max_pending
        )
        
        self.critic = Critic(
            hidden_dim=hidden_dim,
            global_dim=global_dim
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, 
                observation: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None,
                deterministic: bool = False,
                epsilon: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        完整前向传播
        
        Args:
            observation: {
                'uav_states': [batch, num_uavs, 6],
                'pending_requests': [batch, max_pending, 5],
                'global_features': [batch, 4]
            }
            masks: 动作掩码
            deterministic: 是否确定性选择
            epsilon: ε-贪心探索概率，用于缓解无效动作死锁
        
        Returns:
            包含action、value、log_prob等的字典
        """
        uav_states = observation['uav_states']
        request_states = observation['pending_requests']
        global_features = observation['global_features']
        
        # 编码
        uav_embeds = self.uav_encoder(uav_states)  # [batch, num_uavs, hidden_dim]
        request_embeds = self.request_encoder(request_states)  # [batch, max_pending, hidden_dim]
        
        # 可选的交叉注意力
        if self.use_cross_attn:
            # Request作为Query，UAV作为Key/Value
            request_embeds, _ = self.cross_attn(
                query=request_embeds,
                key_value=uav_embeds
            )
        
        # Critic：计算状态价值
        value = self.critic(uav_embeds, request_embeds, global_features)
        
        # Actor：选择动作（支持 ε-贪心探索）
        action, action_info = self.actor.get_action(
            uav_embeds, request_embeds, masks, deterministic, epsilon
        )
        
        return {
            'action': action,
            'value': value,
            'log_prob': action_info['log_prob'],
            'entropy': action_info['entropy'],
            'uav_embeds': uav_embeds,
            'request_embeds': request_embeds,
        }
    
    def evaluate(self,
                 observation: Dict[str, torch.Tensor],
                 actions: Dict[str, torch.Tensor],
                 masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        评估给定动作（用于PPO更新）
        
        Args:
            observation: 观察状态
            actions: 要评估的动作
            masks: 动作掩码
        
        Returns:
            {
                'value': [batch, 1],
                'log_prob': [batch],
                'entropy': [batch]
            }
        """
        uav_states = observation['uav_states']
        request_states = observation['pending_requests']
        global_features = observation['global_features']
        
        # 编码
        uav_embeds = self.uav_encoder(uav_states)
        request_embeds = self.request_encoder(request_states)
        
        # 可选的交叉注意力
        if self.use_cross_attn:
            request_embeds, _ = self.cross_attn(
                query=request_embeds,
                key_value=uav_embeds
            )
        
        # 计算价值
        value = self.critic(uav_embeds, request_embeds, global_features)
        
        # 评估动作
        action_eval = self.actor.evaluate_actions(
            uav_embeds, request_embeds, actions, masks
        )
        
        return {
            'value': value,
            'log_prob': action_eval['log_prob'],
            'entropy': action_eval['entropy']
        }
    
    def get_value(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        只获取状态价值（用于GAE计算）
        
        Args:
            observation: 观察状态
        
        Returns:
            value: [batch, 1]
        """
        uav_states = observation['uav_states']
        request_states = observation['pending_requests']
        global_features = observation['global_features']
        
        uav_embeds = self.uav_encoder(uav_states)
        request_embeds = self.request_encoder(request_states)
        
        if self.use_cross_attn:
            request_embeds, _ = self.cross_attn(
                query=request_embeds,
                key_value=uav_embeds
            )
        
        return self.critic(uav_embeds, request_embeds, global_features)
