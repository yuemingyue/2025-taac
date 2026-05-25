# 2025-taac
初赛26-复赛31
上分尝试：
Base：epo1：0.0163-----------epo3:：0.0230
Batch内对每个正样本选取其余8个正样本作为负样本，使用infonceloss：0.0329
将batch内所有负样本计算infonceloss：0.0429
Scale up 至128bs，64-4-4：0.0461
把测试集的-1去掉，用全部序列推理：0.0535
Consine_warm_up+adamw+rmsnorm调参：0.0570
更改ffn中cnn为linear引入残差结构+moe+去掉dnn后的激活函数：0.0610
扩大bs+全量训练：0.0643
时间差分特征+点击特征+rope编码+加大scale：0.0753
更换初始化以及可学习温度：0.0807
固定温度为+调参+数据增强：0.0843
修改时间特征，新增hour和week，先过emb再过linear后拼接：0.0924
Scale up至128的emb-256的hidden，8-8（block-head）+调参+weighted_loss：0.0979
流行性采样+小改模型架构+修正pos：0.1009
修改代码bug+nxt act特征：0.1059
后处理过滤已经曝光过的item：0.1156

掉分尝试：
直接更换其余多模态特征
多模态融合
迭代困难负样本（向负样本池内加入从当前正样本挑选最难的前10个以及上一batch内的最难的前50个）
自回归训练
Hstu架构
增加更多的norm
Bprloss
Rqvae，语义id
引入点击位置辅助训练
曝光点击特征
Swa融合
LIGR架构
相似度采样
