import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from dataset import MyTestDataset, save_emb
from model import BaselineModel

def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('USER_CACHE_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 保存候选库的embedding和sid
    candidate_ids,candidate_embs= model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    
    return retrieve_id2creative_id,candidate_embs.reshape(candidate_embs.shape[0],candidate_embs.shape[1]),candidate_ids.flatten() 


def compute_recall_k(true_item, rec_items, k=10):
    """
    计算单个用户的 Recall@K（单正样本场景 = HitRate@K）
    """
    rec_set = set(rec_items[:k])
    return 1.0 if true_item in rec_set else 0.0

def compute_ndcg_k(true_item, rec_items, k=10):
    """
    计算单个用户的 NDCG@K（单正样本，binary relevance）
    """
    # 找真实 item 在推荐列表中的位置（从 1 开始）
    try:
        rank = rec_items.index(true_item) + 1  # 位置从 1 开始
        if rank > k:
            return 0.0
        # DCG = 1 / log2(rank + 1)
        dcg = 1.0 / np.log2(rank + 1)
        # IDCG = 1 / log2(1 + 1) = 1.0 （理想情况排第一）
        idcg = 1.0 / np.log2(2)
        return dcg / idcg
    except ValueError:
        # 未命中
        return 0.0

        
def infer(test_loader):
    all_embs = []
    user_list = []
    gt_list=[]
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        seq, token_type,action_type,next_action_type, seq_feat, user_id,gt = batch
        print(seq)
        break
        seq = seq.to(args.device)
        action_type = action_type.to(args.device)
        next_action_type = next_action_type.to(args.device)
        with torch.no_grad():
            logits = model.predict(seq, seq_feat, token_type,action_type,next_action_type)
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        gt_list+=gt
        user_list += user_id
        if step%300==0:
            print(step)
        if step>10:
            break


    # 生成候选库的embedding 以及 id文件
    retrieve_id2creative_id,candidate_embs,candidate_ids = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    all_embs = np.concatenate(all_embs, axis=0)

    query_embs = all_embs  # [B, D]
    topk = 10
    device = torch.device(args.device)

    # 转为 Tensor
    candidate_embs = torch.tensor(candidate_embs, dtype=torch.float32, device=device)  # [N, D]
    query_embs = torch.tensor(query_embs, dtype=torch.float32, device=device)  # [B, D]

    print(f"Starting batched top-{topk} cosine search on {device}...")

    topk_indices_list = []
    BATCH_SIZE = 64  # 可根据显存调整

    for i in tqdm(range(0, query_embs.shape[0], BATCH_SIZE), desc="Query Batch"):
        q_batch = query_embs[i:i+BATCH_SIZE]  # [B', D]

        # 计算余弦相似度（点积）
        sims = q_batch @ candidate_embs.T  # [B', N]

        # 取 top-10 最相似（最大相似度）
        _, topk_idx = torch.topk(sims, k=topk, dim=1, largest=True, sorted=True)  # [B', 10]

        topk_indices_list.append(topk_idx.cpu())

    # 合并
    topk_indices = torch.cat(topk_indices_list, dim=0)  # [num_queries, 10]
    topk_indices = topk_indices.numpy()  # int64

    # 映射到 retrieval_id
    top10s_retrieved = candidate_ids[topk_indices]  # [num_queries, 10]
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    k = 10
    
    recalls = []
    ndcgs = []
    true=0
    for i in range(len(gt_list)):
        true_item = gt_list[i]
        rec_list = top10s[i]
        r = compute_recall_k(true_item, rec_list, k=10)
        if r==1.0:
            true+=1
        n = compute_ndcg_k(true_item, rec_list, k=10)
    
        recalls.append(r)
        ndcgs.append(n)

    avg_recall = np.mean(recalls)
    avg_ndcg = np.mean(ndcgs)

    
    print(f"✅ 用户数: {len(gt_list)}")
    print(f"📊 平均 Recall@10: {avg_recall}")
    print(f"📊 平均 NDCG@10:  {avg_ndcg}")
    print("score:",0.69*avg_ndcg+0.31*avg_recall)
    print("召回个数:",true)

import json
import pickle
import struct
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
import time
import math

class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """
    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        #item id为字符串类型
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        print(self.itemnum,self.usernum)
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer
        #初始化特征信息
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

        self.USER_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = self.feature_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = self.feature_types['item_continual']
        self.USER_ARRAY_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in self.feature_types['item_emb']}  # 记录的是不同多模态特征的维度

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = None
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据
    
        Args:
            uid: 用户ID(reid)
    
        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        with open(self.data_dir / "seq.jsonl", 'rb') as f:
            f.seek(self.seq_offsets[uid])
            line = f.readline()
            data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列
        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t
        
    def _get_diff_cut_edges(self,user_sequence):
        '''
        input:
        user_sequence
        return:
        cut_result: (seq,) 分桶结果
        '''
        # 获取time_stamps_dif:取最后一列.diff.fillna.values
        diff_df = pd.DataFrame(user_sequence).iloc[:,-1].diff().fillna(0).values
        
        # 业务分桶边界（单位: 秒）
        bins = []
        
        # 1) 小时间段
        bins += [30, 60, 300, 600, 900, 1200, 1500, 1800]   # 0.5min, 1min, 5,10,15,20,25,30min
        bins += [3600, 5400, 7200, 9000, 10800, 12600, 14400]  # 1h,1.5h,2h,2.5h,3h,3.5h,4h
        
        # 2) 继续扩展到 10h（36000s），每 0.5h = 1800s 一档
        bins += list(np.arange(16200, 36000 + 1800, 1800))
        
        # 排序 + 去重
        bins = sorted(set(bins))
        
        # 两端加上 -inf, inf
        edges = [-np.inf] + bins + [np.inf]
        # 进行分桶
        cut_result = pd.cut(
            diff_df,
            bins=edges,
            labels=False,   # 返回整数编码
            right=True,     # 区间右闭
            include_lowest=True
        ).astype('int64')
        
        return cut_result
    
    def _get_sparse_time_feat(self,user_sequence):
        '''
        week,day,hour,minutes
        '''
        time_sparse_dict = {}
        time_df = pd.DataFrame(user_sequence)
        
        time_df.columns =  ['user_id','item_id','user_feat','item_feat','action_type','time_stamps']
        # print(time_df)
        time_df['time'] = pd.to_datetime(time_df['time_stamps'],unit = 's')
        time_df['week'] = time_df['time'].dt.weekday
        time_df['hour'] = time_df['time'].dt.hour
        time_sparse_dict['week'] = time_df['week'].values+1
        time_sparse_dict['hour'] = time_df['hour'].values+1
        return time_sparse_dict
        
    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
        # 获取diff 分桶
        diff_cut = self._get_diff_cut_edges(user_sequence)
        time_sparse_dict = self._get_sparse_time_feat(user_sequence)
        
        ext_user_sequence = []
        for seq_index,record_tuple in enumerate(user_sequence):
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            
            if u and user_feat:
                # 添加diff_cut
                user_feat['diff_cut'] = diff_cut[seq_index]
                for time_sparse in time_sparse_dict:
                    user_feat[time_sparse] = time_sparse_dict[time_sparse][seq_index]
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
                
            if i and item_feat:
                item_feat['diff_cut'] = diff_cut[seq_index]    
                for time_sparse in time_sparse_dict:
                    item_feat[time_sparse] = time_sparse_dict[time_sparse][seq_index]
                ext_user_sequence.append((i, item_feat, 1, action_type))
                
        
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = [{} for _ in range(self.maxlen + 1)]
        pos_feat = [{} for _ in range(self.maxlen + 1)]
        neg_feat = [{} for _ in range(self.maxlen + 1)]

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            
            if act_type is not None:
               action_type[idx] = act_type+1
                
            if next_act_type is not None:
                next_action_type[idx] = next_act_type+1
                
            seq_feat[idx] = feat
            if next_i==0:
              print("有0")
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break
              
        # Convert features to tensors
        seq_feat_tensor = self.feat_list_to_tensor(seq_feat)
        pos_feat_tensor = self.feat_list_to_tensor(pos_feat)
        neg_feat_tensor = self.feat_list_to_tensor(neg_feat)

        return seq, pos, neg, token_type, action_type, next_token_type, next_action_type, seq_feat_tensor, pos_feat_tensor, neg_feat_tensor
    
    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        self.item_sparse={'100': 3, '117': 19, '118': 1522, '101': 5, 
        '102': 40125, '114': 20, '112': 28, 
        '121': 4455844, '115': 860, '122': 881, '116': 18,'119':0,'120':0}
        
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []
        feat_types['time_feat'] = ['diff_cut','week','hour']
        
        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = self.item_sparse[feat_id]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )
        for feat_id in feat_types['time_feat']:
            feat_default_value[feat_id] = 0 
            
        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    def feat_list_to_tensor(self, feat_list):
        """
        将特征字典列表转换为 tensor 字典
        """
        feat_tensor_dict = {}
        
        # 收集所有特征ID
        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)

        for feat_id in all_feat_ids:
            if feat_id in getattr(self, 'ITEM_ARRAY_FEAT', {}) or feat_id in getattr(self, 'USER_ARRAY_FEAT', {}):
                # 对于数组特征，需要padding到相同长度
                max_array_len = 0
                max_seq_len = len(feat_list)

                # 计算最大数组长度
                for item in feat_list:
                    if feat_id in item and isinstance(item[feat_id], list):
                        max_array_len = max(max_array_len, len(item[feat_id]))

                # 如果没有数组特征，使用默认值
                if max_array_len == 0:
                    max_array_len = 1  # 至少保持1的维度

                tensor_data = np.zeros((max_seq_len, max_array_len), dtype=np.int64)

                # 填充数据
                for i, item in enumerate(feat_list):
                    if feat_id in item and isinstance(item[feat_id], list):
                        actual_len = min(len(item[feat_id]), max_array_len)
                        tensor_data[i, :actual_len] = item[feat_id][:actual_len]
                    else:
                        # 使用默认值填充
                        default_val = self.feature_default_value.get(feat_id, [0])
                        if isinstance(default_val, list) and len(default_val) > 0:
                            actual_len = min(len(default_val), max_array_len)
                            tensor_data[i, :actual_len] = default_val[:actual_len]

                feat_tensor_dict[feat_id] = torch.from_numpy(tensor_data)
            else:
                # 对于非数组特征，优化tensor创建
                tensor_data = []
                for item in feat_list:
                    default_value = self.feature_default_value.get(feat_id, 0)
                    value = item.get(feat_id, default_value)
                    tensor_data.append(value)

                # 优化：先转换为numpy array再创建tensor
                if feat_id in getattr(self, 'ITEM_SPARSE_FEAT', {}) or feat_id in getattr(self, 'USER_SPARSE_FEAT', {}):
                    # 对于sparse特征使用int类型
                    np_array = np.array(tensor_data, dtype=np.int64)
                else:
                    # 对于continual特征使用float类型
                    np_array = np.array(tensor_data, dtype=np.float32)

                feat_tensor_dict[feat_id] = torch.from_numpy(np_array)

        return feat_tensor_dict
        
    @staticmethod
    def collate_fn(batch):
        seq, pos, neg, token_type,action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        action_type = torch.from_numpy(np.array(action_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))

        
        def merge_feat_tensor(feat_batch):
            if not feat_batch or not feat_batch[0]:
                return {}
            merged = {}
            feat_keys = feat_batch[0].keys()

            for key in feat_keys:
                # 收集所有样本的该特征
                tensors = [fb[key] for fb in feat_batch]

                # 检查是否需要padding（主要是数组特征）
                if tensors[0].dim() == 2:  # 数组特征 [seq_len, array_len]
                    # 找到最大的第二维度
                    max_second_dim = max(tensor.size(1) for tensor in tensors)
                    max_first_dim = max(tensor.size(0) for tensor in tensors)

                    # padding到相同大小
                    padded_tensors = []
                    for tensor in tensors:
                        current_first, current_second = tensor.size()
                        if current_first < max_first_dim or current_second < max_second_dim:
                            pad_tensor = torch.zeros(max_first_dim, max_second_dim, dtype=tensor.dtype)
                            pad_tensor[:current_first, :current_second] = tensor
                            padded_tensors.append(pad_tensor)
                        else:
                            padded_tensors.append(tensor)

                    merged[key] = torch.stack(padded_tensors, dim=0)
                else:  # 一维特征 [seq_len]
                    # 找到最大的维度
                    max_dim = max(tensor.size(0) for tensor in tensors)

                    # padding到相同大小
                    padded_tensors = []
                    for tensor in tensors:
                        current_dim = tensor.size(0)
                        if current_dim < max_dim:
                            if tensor.dtype == torch.long:
                                pad_tensor = torch.zeros(max_dim, dtype=tensor.dtype)
                            else:
                                pad_tensor = torch.zeros(max_dim, dtype=tensor.dtype)
                            pad_tensor[:current_dim] = tensor
                            padded_tensors.append(pad_tensor)
                        else:
                            padded_tensors.append(tensor)

                    merged[key] = torch.stack(padded_tensors, dim=0)

            return merged
        seq_feat = merge_feat_tensor(seq_feat)
        pos_feat = merge_feat_tensor(pos_feat)
        neg_feat = merge_feat_tensor(neg_feat)

        return seq, pos, neg, token_type, action_type,next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class ValidDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)
        self._load_data_and_offsets()
        with open(Path(self.data_dir, 'user_action_type.json'), 'rb') as f:
            self.user_action_type = json.load(f)

    def _load_data_and_offsets(self):
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)
            
    def _load_user_data(self, uid):
        # 每次都重新打开文件，避免多进程问题
        with open(self.data_dir / "seq.jsonl", 'rb') as f:
            f.seek(self.seq_offsets[uid])
            line = f.readline()
            data = json.loads(line)
        return data

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat
    
    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据
        # 获取diff 分桶
        diff_cut = self._get_diff_cut_edges(user_sequence)
        time_sparse_dict = self._get_sparse_time_feat(user_sequence)
        
        ext_user_sequence = []
        for seq_index,record_tuple in enumerate(user_sequence):
        
            u, i, user_feat, item_feat, act_type, _ = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
                    
                
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                #添加特征
                user_feat['diff_cut'] = diff_cut[seq_index]
                for time_sparse in time_sparse_dict:
                    user_feat[time_sparse] = time_sparse_dict[time_sparse][seq_index]
                ext_user_sequence.insert(0, (u, user_feat, 2,act_type))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                
                item_feat['diff_cut'] = diff_cut[seq_index]
                for time_sparse in time_sparse_dict:
                    item_feat[time_sparse] = time_sparse_dict[time_sparse][seq_index]
  
                ext_user_sequence.append((i, item_feat, 1,act_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = [{} for _ in range(self.maxlen + 1)]  # 修改为列表形式
        action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
       
        idx = self.maxlen
        nxt = ext_user_sequence[-1]
        gt = ext_user_sequence[-1][0]
        gt = self.indexer_i_rev[gt]
        # 填充序列数据
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_,act_type_ = record_tuple

            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            
            if act_type_ is not None:
              action_type[idx] = act_type_+1
            if next_act_type is not None:
                next_action_type[idx] = next_act_type+1
            
            seq_feat[idx] = feat
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        # Convert features to tensors
        # 最后置1
        if nxt[3] is not None:
            next_action_type[-1] = nxt[3]+1
        seq_feat_tensor = self.feat_list_to_tensor(seq_feat)

        return seq, token_type,action_type,next_action_type, seq_feat_tensor, user_id,gt

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, dict形式，每个key对应一个tensor
            user_id: user_id, str
        """
        seq, token_type,action_type,next_action_type, seq_feat, user_id,gt = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        action_type = torch.from_numpy(np.array(action_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        def merge_feat_tensor(feat_batch):
            merged = {}
            if feat_batch and feat_batch[0]:  # 检查是否为空
                feat_keys = feat_batch[0].keys()

            for key in feat_keys:
                # 收集所有样本的该特征
                tensors = [fb[key] for fb in feat_batch]
                # merged[key] = torch.stack(tensors, dim=0)
                # 检查是否需要padding（主要是数组特征）
                if tensors[0].dim() == 2:  # 数组特征 [seq_len, array_len]
                    # 找到最大的第二维度
                    max_second_dim = max(tensor.size(1) for tensor in tensors)
                    max_first_dim = max(tensor.size(0) for tensor in tensors)

                    # padding到相同大小
                    padded_tensors = []
                    for tensor in tensors:
                        current_first, current_second = tensor.size()
                        if current_first < max_first_dim or current_second < max_second_dim:
                            pad_tensor = torch.zeros(max_first_dim, max_second_dim, dtype=tensor.dtype)
                            pad_tensor[:current_first, :current_second] = tensor
                            padded_tensors.append(pad_tensor)
                        else:
                            padded_tensors.append(tensor)

                    merged[key] = torch.stack(padded_tensors, dim=0)
                else:  # 一维特征 [seq_len]
                    # 找到最大的维度
                    max_dim = max(tensor.size(0) for tensor in tensors)

                    # padding到相同大小
                    padded_tensors = []
                    for tensor in tensors:
                        current_dim = tensor.size(0)
                        if current_dim < max_dim:
                            if tensor.dtype == torch.long:
                                pad_tensor = torch.zeros(max_dim, dtype=tensor.dtype)
                            else:
                                pad_tensor = torch.zeros(max_dim, dtype=tensor.dtype)
                            pad_tensor[:current_dim] = tensor
                            padded_tensors.append(pad_tensor)
                        else:
                            padded_tensors.append(tensor)

                    merged[key] = torch.stack(padded_tensors, dim=0)

            return merged
       

        seq_feat = merge_feat_tensor(seq_feat)

        return seq, token_type,action_type,next_action_type, seq_feat, user_id,gt


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict


   
