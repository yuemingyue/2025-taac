import argparse
import json
import os
import struct
from pathlib import Path
import time
import pickle
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

def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 保证 cudnn 可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(3407)
def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=8,type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', default=True)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    item_sparse={'100': 3, '117': 19, '118': 1522, '101': 5, 
        '102': 40125, '114': 20, '112': 28, 
        '121': 4455844, '115': 860, '122': 881, '116': 18,'119':0,'120':0}
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(item_sparse[feat_id])
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = item_sparse[feat_id]
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
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else random.randint(1, 8000000)
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
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id

def read_fbin(filepath):
    """
    读取 .fbin 文件：float32 格式，前 8 字节为 (n, d)，接着是 n*d 的数据
    """
    with open(filepath, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.float32, count=n * d)
        return data.reshape(n, d)


def read_u64bin(filepath):
    with open(filepath, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        assert d == 1, f"d should be 1, got {d}"
        ids = np.fromfile(f, dtype=np.uint64, count=n)
        return ids
def infer():
    seed_everything(3407)
    args = get_args()
    args.diy_hidden_units={'item_embed_size':64,'user_embed_size':64,'pos_embed_size':768,'user_dnn_size':768,'item_dnn_size':768,'item_mul_embed_size':768}
    args.feature_embedding_dims={
    '103': 16,'104': 16,'105': 16,'109': 16,'100': 16,'117': 16,
    '118': 32,'101': 16,'102': 32,'114': 16,'112': 16,
    '121': 32,'115': 16,'122': 32,'116': 16,'106': 16,'107': 16,'108': 16,'110': 16 ,'119':16,'120':16
    }
    data_path = os.environ.get('EVAL_DATA_PATH')

    # 构建用户历史交互字典
    cache_dir = Path(os.environ.get('USER_CACHE_PATH'))
    user_seq_file = cache_dir / "user_his_act.pkl"
    t1=time.time()
    with open(user_seq_file, "rb") as f:
        user_his_act = pickle.load(f)

        
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=test_dataset.collate_fn,pin_memory=True
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    all_embs = []
    user_list = []
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

        seq, token_type,action_type,next_action_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        action_type = action_type.to(args.device)
        next_action_type = next_action_type.to(args.device)
        with torch.no_grad():
            logits = model.predict(seq, seq_feat, token_type,action_type,next_action_type)
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id
        if step%300==0:
            print(step)





    # 生成候选库的embedding 以及 id文件
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    all_embs = np.concatenate(all_embs, axis=0)
    indexer = test_dataset.indexer['i']
    # 保存query文件
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))
    query_emb_path = Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin')
    save_emb(all_embs, query_emb_path)

    # 加载候选 embedding 和 retrieval_id
    candidate_emb_path = Path(os.environ.get('EVAL_RESULT_PATH'), 'embedding.fbin')
    candidate_id_path = Path(os.environ.get('EVAL_RESULT_PATH'), 'id.u64bin')

    print("Loading candidate embeddings and ids...")
    candidate_embs = read_fbin(candidate_emb_path)  # [N, D]
    candidate_ids = read_u64bin(candidate_id_path).flatten()  # [N,]

    assert len(candidate_embs) == len(candidate_ids), "Mismatch: embs and ids count"

    query_embs = all_embs  # [B, D]
    topk = 30
    device = torch.device(args.device)

    # 转为 Tensor
    candidate_embs = torch.tensor(candidate_embs, dtype=torch.float32, device=device)  # [N, D]
    query_embs = torch.tensor(query_embs, dtype=torch.float32, device=device)  # [B, D]

    print(f"Starting batched top-{topk} cosine search on {device}...")

    topk_indices_list = []
    BATCH_SIZE = 256  # 可根据显存调整

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
    #过滤
    fliter_top10s_retrieved=[]
    num=0
    for i in range(len(top10s_retrieved)):
        fliter_re_ids = []
        fliter_out_re_ids =[]
        topk_re=top10s_retrieved[i].tolist()
        user_id=user_list[i]
        user_act=list(user_his_act.get(user_id,set()))
        #进行过滤操作
        for r_id in topk_re:
            creative_id = retrieve_id2creative_id.get(int(r_id), 0)
            item_id = indexer[creative_id] if creative_id in indexer else 0
            if item_id not in user_act:
                fliter_re_ids.append(r_id)
            else:
                num+=1
                fliter_out_re_ids.append(r_id)
        if len(fliter_re_ids)>10:
            fliter_top10s_retrieved.append(fliter_re_ids[:10])
        else:
            fliter_re_ids=fliter_re_ids+fliter_out_re_ids[:10-len(fliter_re_ids)]
            fliter_top10s_retrieved.append(fliter_re_ids[:10])
            
    fliter_top10s_retrieved=np.array(fliter_top10s_retrieved)
    top10s_untrimmed = []
    for top10 in tqdm(fliter_top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]
    print("过滤总数",num)
    print("平均每个用户过滤个数",num/len(top10s))
    
    return top10s, user_list


   
