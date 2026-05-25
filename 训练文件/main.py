import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
import json
import pandas as pd
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from dataset import MyDataset,MyTestDataset,save_emb
from model import BaselineModel,RMSNorm
import random
import numpy as np
import torch
import torch.nn.functional as F
from infer import process_cold_start_feat,get_candidate_emb,compute_recall_k,compute_ndcg_k,ValidDataset
from torch.cuda.amp import autocast, GradScaler


def init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='selu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_embed(m):
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if hasattr(m, 'padding_idx') and m.padding_idx is not None:
            m.weight.data[m.padding_idx].zero_()

def init_norm(m):
    if isinstance(m, (RMSNorm, nn.LayerNorm)):
        if hasattr(m, 'scale'):
            nn.init.ones_(m.scale)
        elif hasattr(m, 'weight'):
            nn.init.ones_(m.weight)

def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=8,type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
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

def infer(test_loader,model):
    model.eval()
    all_embs = []
    user_list = []
    gt_list=[]
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        seq, token_type,action_type,next_action_type, seq_feat, user_id,gt = batch
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

    for i in range(0, query_embs.shape[0], BATCH_SIZE):
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

    return avg_recall,avg_ndcg,0.69*avg_ndcg+0.31*avg_recall
    

        
if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')
    seed_everything(3407)
    args = get_args()
    args.diy_hidden_units={'item_embed_size':64,'user_embed_size':64,'pos_embed_size':768,'user_dnn_size':768,'item_dnn_size':768,'item_mul_embed_size':768}
    args.feature_embedding_dims={
    '103': 16,'104': 16,'105': 16,'109': 16,'100': 16,'117': 16,
    '118': 32,'101': 16,'102': 32,'114': 16,'112': 16,
    '121': 32,'115': 16,'122': 32,'116': 16,'106': 16,'107': 16,'108': 16,'110': 16 ,'119':16,'120':16
    }
    dataset = MyDataset(data_path, args)
    train_dataset = dataset
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=dataset.collate_fn,pin_memory=True
    )

    test_dataset = ValidDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=test_dataset.collate_fn,pin_memory=True
    )
    
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    

    model.apply(init_linear)
    model.apply(init_embed)
    model.apply(init_norm)

    # ID embedding 零初始化（可选）
    model.item_emb.weight.data[model.item_emb.padding_idx].zero_()
    model.user_emb.weight.data[model.user_emb.padding_idx].zero_()
    model.pos_emb.weight.data[model.pos_emb.padding_idx].zero_()

       
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = 0.1*total_steps
    scheduler = CosineLRScheduler(optimizer, t_initial=total_steps, warmup_t=warmup_steps, lr_min=0, warmup_lr_init=0, t_in_epochs=False)

    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")

    optimizer.zero_grad()  # 初始化梯度

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        train_pos_sim = 0
        train_neg_sim = 0
        train_gap_sim = 0
        optimizer.zero_grad()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            step+=1
            seq, pos, neg, token_type,action_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            action_type = action_type.to(args.device)
            next_action_type = next_action_type.to(args.device)

            with autocast( dtype=torch.bfloat16):
                seq_embs, pos_embs,neg_embs,loss_mask = model(
                seq, pos, neg, token_type, action_type,next_token_type, next_action_type, seq_feat, pos_feat, neg_feat)
                
                hidden_size = neg_embs.size(-1)
                loss_mask = loss_mask.bool()
                seq_embs = seq_embs[loss_mask]
                pos_embs = pos_embs[loss_mask]
                neg_embs = neg_embs[loss_mask]
                next_action_type = next_action_type[loss_mask] 
                hidden_size =neg_embs.size(-1)
                # print(seq_emb.shape,pos_embs.shape,neg_embs.shape)
                seq_embs = seq_embs /(seq_embs.norm(dim=-1,keepdim=True)+1e-8)
                pos_embs = pos_embs /(pos_embs.norm(dim=-1,keepdim=True)+1e-8)
                neg_embs =neg_embs /(neg_embs.norm(dim=-1,keepdim=True)+1e-8)
                N, D = pos_embs.shape
                
                #正样本相似度
                pos_logits = F.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1)
                train_pos_sim+=pos_logits.mean().item()
                
                neg_embedding_all = neg_embs.reshape(-1,hidden_size)
                neg_logits = torch.matmul(seq_embs, neg_embedding_all.transpose(-1,-2))
                train_neg_sim+=neg_logits.mean().item()
                
                logits =torch.cat([pos_logits,neg_logits],dim=-1)
                temperature = 0.02
                logits =logits/temperature
                    
                labels = torch.zeros(logits.size(0),device=logits.device, dtype=torch.int64)
                
                #log 加权
                loss = F.cross_entropy(logits, labels, reduction='none')
                transform_weight = torch.tensor(2.5, device=logits.device, dtype=torch.bfloat16)
                click_weight = torch.tensor(1.5, device=logits.device, dtype=torch.bfloat16)
                expose_weight = torch.tensor(1.0, device=logits.device, dtype=torch.bfloat16)
                
                weights = torch.zeros_like(next_action_type, dtype=torch.bfloat16)  # 默认权重 1.0
    
                weights = torch.where(next_action_type == 3, transform_weight, weights)
                weights = torch.where(next_action_type == 2, click_weight, weights)
                weights = torch.where(next_action_type == 1, expose_weight, weights)
                loss = (loss * weights).mean()

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)
            
            
            if step%200==0:
                writer.add_scalar("train_sim/pos_sim", train_pos_sim/step ,global_step)
                writer.add_scalar("train_sim/neg_sim", train_neg_sim/step, global_step)
                writer.add_scalar('train_sim/sim_gap', train_pos_sim/step-train_neg_sim/step, global_step)

            writer.add_scalar('Loss/train', loss.item(), global_step)
           
            # 混合精度反向传播
            loss.backward()
            # 梯度裁剪
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)     
            writer.add_scalar('grad_norm', grad_norm, global_step)
            
            # 优化器更新
            optimizer.step()


        
            scheduler.step_update(num_updates=global_step)
            optimizer.zero_grad()  # 清空累积的梯度

            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
            
                

        #去掉验证函数
        #recall,ndcg,score = infer(test_loader,model)

        #修改模型名字
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
