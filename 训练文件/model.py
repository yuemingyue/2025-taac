from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset import save_emb
import math
import torch.nn.functional as F
class AxialPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, axial_dim_1, axial_dim_2):
        super(AxialPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.axial_dim_1 = axial_dim_1
        self.axial_dim_2 = axial_dim_2

        # 定义两个轴向的位置嵌入
        self.axial_wpe_1 = nn.Parameter(torch.randn(axial_dim_1, embedding_dim) * 0.01)
        self.axial_wpe_2 = nn.Parameter(torch.randn(axial_dim_2, embedding_dim) * 0.01)

        # 缓存组合后的 wpe，避免重复计算
        #self._wpe = None

    def forward(self,x):
        bs,seq,dim = x.shape

        # 扩展维度并广播相加
        axial_wpe_1 = self.axial_wpe_1.unsqueeze(1).expand(-1, self.axial_dim_2, self.embedding_dim)
        axial_wpe_2 = self.axial_wpe_2.unsqueeze(0).expand(self.axial_dim_1, -1, self.embedding_dim)

        # 组合两个轴向嵌入
        wpe = (axial_wpe_1 + axial_wpe_2) / 2

        # 展平为 (axial_dim_1 * axial_dim_2, embedding_dim)
        wpe = wpe.view(self.axial_dim_1 * self.axial_dim_2, self.embedding_dim)
        wpe = wpe.unsqueeze(0).expand(bs,-1,-1)
        x += wpe
        return x
class MoEClassifier(nn.Module):
    def __init__(self, num_experts, input_dim, num_classes):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, num_classes)
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)  
        # 门控权重归一化：每个位置的专家权重之和为 1
        gate_weights = F.softmax(gate_logits, dim=-1)  # 形状：(b, l, num_experts)
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)  
        gate_weights = gate_weights.permute(0, 2, 1).unsqueeze(-1)  # 维度转换：(b, l, num_experts) → (b, num_experts, l, 1)
        fused_logits = (gate_weights * expert_logits).sum(dim=1)  # 最终形状：(b, l, num_classes)
        return fused_logits

def rotate_half(x):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, max_len: int = 4096, device=None, dtype=torch.float32):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE: head_dim 必须是偶数。"
        self.head_dim = head_dim
        self.base = base
        self.max_len = max_len
        self.dtype = dtype

        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

        # 先构建一次缓存
        self._build_cache(max_len, device=device, dtype=dtype)

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device=None, dtype=None):
        device = device if device is not None else (self.cos_cached.device if self.cos_cached is not None else "cpu")
        dtype = dtype if dtype is not None else self.dtype

        half_dim = self.head_dim // 2
        # 频率：base ^ (-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half_dim, device=device, dtype=dtype) / half_dim))  # [Dh/2]
        t = torch.arange(seq_len, device=device, dtype=dtype)  # [T]
        freqs = torch.einsum("t,f->tf", t, inv_freq)  # [T, Dh/2]
        emb = torch.cat([freqs, freqs], dim=-1)       # [T, Dh]
        cos = emb.cos()[None, None, ...]              # [1, 1, T, Dh]
        sin = emb.sin()[None, None, ...]              # [1, 1, T, Dh]
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(self, seq_len: int, device=None, dtype=None):
        # 动态扩容
        need_rebuild = (
            self.cos_cached is None
            or seq_len > self.cos_cached.shape[2]
            or (device is not None and self.cos_cached.device != device)
            or (dtype is not None and self.cos_cached.dtype != dtype)
        )
        if need_rebuild:
            self._build_cache(max(seq_len, self.max_len), device=device, dtype=dtype)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        # 仍保持广播形状 [1, 1, T, Dh]
        return cos, sin

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 计算均方根
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)

class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate,args):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = 0

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)
        
        self.rope = RotaryEmbedding(self.head_dim, base= 10000.0, max_len=args.maxlen+1,device = args.device)

    
    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = Q.shape[-2]

        # 取 cos/sin 并应用 RoPE
        cos, sin = self.rope(seq_len=kv_seq_len, device=Q.device, dtype=Q.dtype)  # [1,1,T,Dh]4
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        output = self.out_linear(attn_output)
        return output, None



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, expansion_factor=4):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_units, hidden_units * expansion_factor)
        self.linear2 = torch.nn.Linear(hidden_units * expansion_factor, hidden_units)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(hidden_units)

    def forward(self, inputs):
        # inputs: (batch, seq_len, hidden_units)
        x = self.linear1(inputs)           # (batch, seq_len, expansion)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)                # (batch, seq_len, hidden_units)
        x = self.dropout(x)
        # Add residual connection and layer norm (common in Transformers)
        return self.layer_norm(x + inputs)  # 改进：加入残差连接


class BaselineModel(torch.nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen

        self.item_emb = torch.nn.Embedding(self.item_num + 1,args.diy_hidden_units['item_embed_size'], padding_idx=0)
        self.item_linear = torch.nn.Linear(args.diy_hidden_units['item_embed_size'],args.diy_hidden_units['item_dnn_size'])
        self.user_emb = torch.nn.Embedding(self.user_num + 1,args.diy_hidden_units['user_embed_size'], padding_idx=0)
        self.user_linear = torch.nn.Linear(args.diy_hidden_units['user_embed_size'],args.diy_hidden_units['user_dnn_size'])
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.diy_hidden_units['pos_embed_size'], padding_idx=0)
        self.axial_emb  = AxialPositionalEmbedding(args.diy_hidden_units['user_dnn_size'],3,34)
        
        self.emb_dropout = torch.nn.Dropout(p=0.3)
        self.sparse_emb = torch.nn.ModuleDict()
        self.sparse_dnn = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        
        # time_diff_cut
        self.diff_cut_emb = torch.nn.Embedding(29,args.diy_hidden_units['user_dnn_size']//2, padding_idx=0)
        self.diff_dnn = torch.nn.Linear(args.diy_hidden_units['user_dnn_size']//2,args.diy_hidden_units['user_dnn_size'])
        
        # time_sparse
        self.week_emb = torch.nn.Embedding(8,args.diy_hidden_units['user_dnn_size']//2, padding_idx=0)
        self.hour_emb = torch.nn.Embedding(25,args.diy_hidden_units['user_dnn_size']//2, padding_idx=0)
        self.time_sparse_dnn = torch.nn.Linear(args.diy_hidden_units['user_dnn_size'],args.diy_hidden_units['user_dnn_size'])
        
        # act emb
        self.act_emb = torch.nn.Embedding(4,args.diy_hidden_units['user_dnn_size']//2, padding_idx=0)
        self.act_dnn = torch.nn.Linear(args.diy_hidden_units['user_dnn_size']//2,args.diy_hidden_units['user_dnn_size'])

        # next act emb
        self.next_act_emb = torch.nn.Embedding(4,args.diy_hidden_units['user_dnn_size']//2, padding_idx=0)
        self.next_act_dnn = torch.nn.Linear(args.diy_hidden_units['user_dnn_size']//2,args.diy_hidden_units['user_dnn_size'])

        
        self.fusionForAtta = MoEClassifier(10,args.diy_hidden_units['user_dnn_size']*4,args.diy_hidden_units['user_dnn_size'])
        
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        #计算用户和物品特征维度
        userdim =  sum([args.feature_embedding_dims[k] for k in self.USER_SPARSE_FEAT])+ \
        sum([args.feature_embedding_dims[k] for k in self.USER_ARRAY_FEAT])+len(self.USER_CONTINUAL_FEAT)+args.diy_hidden_units['user_dnn_size']
        
        itemdim =sum([args.feature_embedding_dims[k] for k in  self.ITEM_SPARSE_FEAT])+ \
        sum([args.feature_embedding_dims[k] for k in self.ITEM_ARRAY_FEAT])+len(self.ITEM_CONTINUAL_FEAT)+ \
        args.diy_hidden_units['item_mul_embed_size'] * len(self.ITEM_EMB_FEAT) + args.diy_hidden_units['item_dnn_size']

        self.user_dnn_size = args.diy_hidden_units['user_dnn_size']
        self.item_dnn_size = args.diy_hidden_units['item_dnn_size']

        self.userdnn = torch.nn.Sequential(
                      torch.nn.Linear(userdim,args.diy_hidden_units['user_dnn_size']),
                      torch.nn.GELU(),
                      RMSNorm(args.diy_hidden_units['user_dnn_size'], eps=1e-8)
                      )
        self.itemdnn = torch.nn.Sequential(
                      torch.nn.Linear(itemdim,args.diy_hidden_units['item_dnn_size']),
                      torch.nn.GELU(),
                      RMSNorm(args.diy_hidden_units['user_dnn_size'], eps=1e-8)
                      )
        
        self.last_layernorm = RMSNorm(args.diy_hidden_units['user_dnn_size'], eps=1e-8)
    
  
        for _ in range(args.num_blocks):
            new_attn_layernorm = RMSNorm(args.diy_hidden_units['user_dnn_size'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.diy_hidden_units['user_dnn_size'], args.num_heads, args.dropout_rate,args = args
            )  
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = RMSNorm(args.diy_hidden_units['user_dnn_size'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.diy_hidden_units['user_dnn_size'], args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
            
         #初始化各个embedding层
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.feature_embedding_dims[k], padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.feature_embedding_dims[k], padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.feature_embedding_dims[k], padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1,  args.feature_embedding_dims[k], padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k],args.diy_hidden_units['item_mul_embed_size'])
    
    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def feat2emb(self, seq, feature_tensor_dict, mask=None, include_user=False):
        seq = seq.to(self.dev)
        # pre-compute embedding
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_linear(self.user_emb(user_mask * seq))
            item_embedding = self.item_linear(self.item_emb(item_mask * seq))
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_linear(self.item_emb(seq))
            item_feat_list = [item_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                try:
                  tensor_feature = feature_tensor_dict[k].to(self.dev)
                except Exception as e:
                    # 如果 try 块中发生任何异常，这里的代码会被执行
                    print(f"处理特征 '{k}' 时发生错误:")
                    print(f"  错误类型: {type(e).__name__}")
                    print(f"  错误信息: {e}")
                if feat_type.endswith('sparse'):
                    tensor_feature = self.sparse_emb[k](tensor_feature)
                    feat_list.append(tensor_feature)
                elif feat_type.endswith('array'):
                    tensor_feature = self.sparse_emb[k](tensor_feature).sum(2)
                    feat_list.append(tensor_feature)
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            tensor_feature = feature_tensor_dict[k].to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # merge features
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = self.itemdnn(all_item_emb)
        
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = self.userdnn(all_user_emb)
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
            
        return seqs_emb

    def log2feats(self, log_seqs, mask, action_type,next_action_type,seq_feature):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        
        
        # 添加diff cut embeding
        diff_cut_embs = self.diff_cut_emb(seq_feature['diff_cut'].long().to(self.dev))
        diff_cut_embs = self.diff_dnn(diff_cut_embs)
        # 添加稀疏时间
        week_emb = self.week_emb(seq_feature['week'].long().to(self.dev))
        hour_emb = self.hour_emb(seq_feature['hour'].long().to(self.dev))
        time_sparse_emb = torch.cat((week_emb,hour_emb),dim = -1)
        time_sparse_emb = self.time_sparse_dnn(time_sparse_emb)
        # 添加action embedding
        #act_embs = self.act_emb(action_type)
        #act_embs = self.act_dnn(act_embs)

        # 添加next action embedding
        next_act_embs = self.next_act_emb(next_action_type)
        next_act_embs = self.next_act_dnn(next_act_embs)

        # 拼接特征 
        seqs = torch.cat((seqs,diff_cut_embs,time_sparse_emb,next_act_embs),dim = -1)
        # 整合送入att
        seqs = self.fusionForAtta(seqs)
        # 修正缩放位置
        seqs *= self.item_emb.embedding_dim**0.5
        
        #添加poss位置
        mask_valid = log_seqs != 0
        poss = mask_valid.cumsum(dim=1)
        seqs += self.pos_emb(poss)
        # 添加轴向位置编码
        seqs = self.axial_emb(seqs)
        seqs = self.emb_dropout(seqs)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    #训练时调用，计算正负样本的logits
    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, action_type,next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
        """
        训练时调用，计算正负样本的logits

        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典

        Returns:
            pos_logits: 正样本logits，形状为 [batch_size, maxlen]
            neg_logits: 负样本logits，形状为 [batch_size, maxlen]
        """
        log_feats = self.log2feats(user_item, mask,action_type, next_action_type,seq_feature)
        loss_mask = (next_mask == 1).to(self.dev)
        # 复赛新增代码：增大转化行为在训练中的权重
        loss_mask = (next_action_type == 3).to(self.dev).float() * 1.5 + loss_mask.float()
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)
        
        return log_feats, pos_embs,neg_embs,loss_mask

    def predict(self, log_seqs, seq_feature, mask,action_type,next_action_type):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        log_feats = self.log2feats(log_seqs, mask, action_type,next_action_type,seq_feature)
        log_feats = log_feats /(log_feats.norm(dim=-1,keepdim=True)+1e-8)

        final_feat = log_feats[:, -1, :]

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            batch_emb = self.save_feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            
            batch_emb =batch_emb /(batch_emb.norm(dim=-1,keepdim=True)+1e-8)
            
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        return final_ids,final_embs

    def save_feat2emb(self, seq, feature_array, mask=None, include_user=False):
        seq = seq.to(self.dev)
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_linear(self.user_emb(user_mask * seq))
            item_embedding = self.item_linear(self.item_emb(item_mask * seq))
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_linear(self.item_emb(seq))
            item_feat_list = [item_embedding]

        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]
        if include_user:
            all_feat_types.extend([
                (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
            ])

        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue
            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)
                if feat_type.endswith('sparse'):
                    tensor_feature = self.sparse_emb[k](tensor_feature)
                    feat_list.append(tensor_feature)
                elif feat_type.endswith('array'):
                    tensor_feature = self.sparse_emb[k](tensor_feature).sum(2)
                    feat_list.append(tensor_feature)
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = self.itemdnn(all_item_emb)
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = self.userdnn(all_user_emb)
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb
        
    def feat2tensor(self, seq_feature, k):
        batch_size = len(seq_feature)
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            max_array_len = max(max(len(item_data) for item_data in [item[k] for item in seq]) for seq in seq_feature)
            max_seq_len = max(len(seq) for seq in seq_feature)
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data).to(self.dev)
        else:
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i, :len(seq_data)] = seq_data
            return torch.from_numpy(batch_data).to(self.dev)