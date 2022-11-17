import os
from itertools import zip_longest
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import pickle
# from seqeval.metrics import f1_score

def build_corpus(split, make_vocab=True, data_dir="final数据集(村镇分离)"):
    # 读取数据
    # 确定split是三种数据集其中之一
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    # 将路径和选择的数据集种类相连作为路径(".char.bmes"应可改为自己需要的)
    with open(os.path.join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []

        # 读取文件内容并去除首尾多余换行
        # 按行读取文件
        for line in f:
            if line != '\n':
                word, tag = line.split()
                # print(word,tag)
                word_list.append(word)
                tag_list.append(tag)
            # 遇到空行便说明这句话结束，在这句话结尾添加<END>存入数组
            else:
                word_lists.append(word_list+["<END>"])
                tag_lists.append(tag_list+["<END>"])
                # 清空
                word_list = []
                tag_list = []
                # print(word_lists,tag_lists)
    # 将语句及其标签按照长度降序排列
    word_lists = sorted(word_lists, key=lambda x: len(x), reverse=True)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=True)

    # 如果make_vocab为True，还需要返回word2index和tag2index
    if make_vocab:
        word2index = build_map(word_lists)
        tag2index = build_map(tag_lists)
        word2index['<UNK>'] = len(word2index)
        word2index['<PAD>'] = len(word2index)
        word2index["<START>"] = len(word2index)
        # word2index["<END>"] = len(word2index)

        tag2index['<PAD>'] = len(tag2index)
        tag2index["<START>"] = len(tag2index)
        # tag2index["<END>"] = len(tag2index)
        return word_lists, tag_lists, word2index, tag2index
    else:
        return word_lists, tag_lists

def build_map(lists):
    maps = {}
    # 按照其位置顺序作为index
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

class MyDataset(Dataset):
    def __init__(self,data,tags,word2index,tag2index):
        self.data = data
        self.tags = tags
        self.word2index = word2index
        self.tag2index = tag2index

    def __getitem__(self,index):
        datum = self.data[index]
        tag  = self.tags[index]

        datum_index = [self.word2index.get(i,self.word2index["<UNK>"]) for i in datum]
        tag_index  = [self.tag2index[i] for i in tag]

        return datum_index,tag_index

    def __len__(self):
        assert len(self.data) == len(self.tags)
        return len(self.tags)

    def pro_batch_data(self,batch_data):
        global device
        data = []
        tags = []
        batch_lens = []

        for datum,tag in batch_data:
            data.append(datum)
            tags.append(tag)
            # 该batch中记录每条语句的长度
            batch_lens.append(len(datum))
        # 取最大语句长度
        batch_max_len = max(batch_lens)
        # 进行填充<PAD>的编码使每条语句长度一致
        data = [i + [self.word2index["<PAD>"]] * (batch_max_len - len(i)) for i in data]
        tags = [i + [self.tag2index["<PAD>"]] * (batch_max_len - len(i)) for i in tags]
        # 返回tensor类型
        return torch.tensor(data,dtype=torch.int64,device=device),torch.tensor(tags,dtype=torch.long,device=device),batch_lens


class Mymodel(nn.Module):
    def __init__(self,corpus_num,embedding_num,hidden_num,class_num,bi=True):
        # 调用父类初始化
        super().__init__()

        # 创建一个简单的存储固定大小的词典的嵌入向量的查找表：输入为一个编号列表，输出为对应的符号嵌入向量列表
        # corpus_num为word2index长度，embedding_num为编码维度，hidden_num为隐藏层矩阵列数
        self.embedding = nn.Embedding(corpus_num,embedding_num)
        self.lstm = nn.LSTM(embedding_num,hidden_num,batch_first=True,bidirectional=bi)

        # 若lstm双向
        if bi:
            # 分类器(线性相乘)
            self.classifier = nn.Linear(hidden_num * 2,class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)
        # 创建一个大小为class_num x class_num的全为1/class_num的转移矩阵
        self.transition = nn.Parameter(torch.ones(class_num, class_num) * 1 / class_num)
        self.loss_fun = self.cal_lstm_crf_loss

    def cal_lstm_crf_loss(self,crf_scores, targets):
        """计算双向LSTM-CRF模型的损失
        该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
        """
        # 计算双向LSTM-CRF模型的损失
        # crf_scores:发射矩阵和转移矩阵的和，targets:batch对应的标签
        global tag2index
        pad_index = tag2index.get('<PAD>')
        start_index = tag2index.get('<START>')
        end_index = tag2index.get('<END>')

        device = crf_scores.device

        # targets:[B(batch_size),L(max_len)],crf_scores:[B,L,T(target set size),T]
        batch_size, max_len = targets.size()
        target_size = len(tag2index)

        # mask = 1 - ((targets == pad_index) + (targets == end_index))  # [B, L]
        # 数组/矩阵掩码(布尔值表示)
        mask = (targets != pad_index)
        lengths = mask.sum(dim=1)
        targets = self.indexed(targets, target_size, start_index)

        # # 计算Golden scores方法１
        # import pdb
        # pdb.set_trace()
        # 按照mask输出tensor，输出为向量
        targets = targets.masked_select(mask)  # [real_L]
        # 将mask改变为(batch_size,max_len,1,1)的维度格式，再扩大为crf_scores相同的大小，然后改变为(-1,targets_size * targets_size)的维度格式
        flatten_scores = crf_scores.masked_select(mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)).view(-1,target_size * target_size).contiguous()
        # 以列形式按照targets在新增一个维度后的顺序存储flatten_scores
        golden_scores = flatten_scores.gather(dim=1, index=targets.unsqueeze(1)).sum()

        # 计算golden_scores方法２：利用pack_padded_sequence函数
        # targets[targets == end_index] = pad_index
        # scores_at_targets = torch.gather(
        #     crf_scores.view(batch_size, max_len, -1), 2, targets.unsqueeze(2)).squeeze(2)
        # scores_at_targets, _ = pack_padded_sequence(
        #     scores_at_targets, lengths-1, batch_first=True
        # )
        # golden_scores = scores_at_targets.sum()

        # 计算all path scores
        # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
        scores_upto_t = torch.zeros(batch_size, target_size).to(device)
        for t in range(max_len):
            # 当前时刻 有效的batch_size(因为有些序列比较短)
            batch_size_t = (lengths > t).sum().item()   # 取出有效序列后求和并使用.item()从tensor格式转化成python的正常数字格式
            if t == 0:
                scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,t, start_index, :]
            else:
                # 我们将当前时间段的分数与截至上一时间段的分数相加，然后进行log-sum-exp。
                # 记住，前一个时间段的cur_tag就是这个时间段的prev_tag。
                # 所以，沿着当前时间段的cur_tag维度广播前一个时间段的cur_tag分数。
                scores_upto_t[:batch_size_t] = torch.logsumexp(
                    crf_scores[:batch_size_t, t, :, :] +
                    scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1
                )
        all_path_scores = scores_upto_t[:, end_index].sum()

        # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
        loss = (all_path_scores - golden_scores) / batch_size
        return loss

    def indexed(self,targets, tagset_size, start_index):
        # 将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类
        batch_size, max_len = targets.size()
        for col in range(max_len - 1, 0, -1):
            # targets[:,col]表示取targets中所有行的第col列
            targets[:, col] += (targets[:, col - 1] * tagset_size)
        targets[:, 0] += (start_index * tagset_size)
        return targets

    def forward(self,batch_data,batch_tag=None):
        embedding = self.embedding(batch_data)
        out,_ = self.lstm(embedding)

        emission = self.classifier(out)
        # 发射矩阵的三个维度
        batch_size, max_len, out_size = emission.size()
        # 增加发射函数的维度并和转移矩阵求和
        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1) + self.transition

        if batch_tag is not None:
            loss = self.cal_lstm_crf_loss(crf_scores,batch_tag)
            return loss
        else:
            return crf_scores

    def test(self, test_sents_tensor, lengths):
        # 使用维特比算法进行解码
        global tag2index
        start_index = tag2index['<START>']
        end_index = tag2index['<END>']
        pad = tag2index['<PAD>']
        tagset_size = len(tag2index)

        crf_scores = self.forward(test_sents_tensor)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()

        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)

        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的index，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_index).to(device)
        lengths = torch.LongTensor(lengths).to(device)

        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_index
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_index, :]
                backpointer[: batch_size_t, step, :] = start_index
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagindexes = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_index
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_index] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagindexes.append(tags_t.tolist())

        # zip_longest具体可以用来对列表的一一对应，
        # 如果列表的长度不一致，则其会选择最长的那个列表，并将没有的填充为None
        tagindexes = list(zip_longest(*reversed(tagindexes), fillvalue=pad))
        tagindexes = torch.Tensor(tagindexes).long()

        # 返回解码的结果
        return tagindexes.reshape(-1)

def test():
    global word2index,model,index2tagzou,device
    while True:
        text = input("请输入：")
        text_index = [[word2index.get(i,word2index["<UNK>"]) for i in text] + [word2index["<END>"]]]

        text_index = torch.tensor(text_index,dtype=torch.int64,device=device)
        pre = model.test(text_index,[len(text)+1])
        pre = [index2tagzou[i] for i in pre]
        print([f'{w}_{s}' for w,s in zip(text,pre)])




if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_data,train_tag,word2index,tag2index = build_corpus("train",make_vocab=True)
    dev_data,dev_tag = build_corpus("dev",make_vocab=False)
    index2tagzou = [i for i in tag2index]

    corpus_num = len(word2index)
    class_num  = len(tag2index)

    print('corpus_num:',corpus_num)
    print('class_num:',class_num)

    epoch = 20
    train_batch_size = 40
    dev_batch_size = 100
    embedding_num = 101
    hidden_num = 128
    bi = True
    lr = 0.0001
    train_dataset = MyDataset(train_data,train_tag,word2index,tag2index)
    train_dataloader = DataLoader(train_dataset,train_batch_size,shuffle=False,collate_fn=train_dataset.pro_batch_data)

    dev_dataset = MyDataset(dev_data, dev_tag, word2index, tag2index)
    dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=False,collate_fn=dev_dataset.pro_batch_data)

    model = Mymodel(corpus_num,embedding_num,hidden_num,class_num,bi)
    opt = torch.optim.AdamW(model.parameters(),lr = lr)
    model = model.to(device)

    for e in range(epoch):
        model.train()
        for batch_data,batch_tag,batch_len in train_dataloader:
            train_loss = model.forward(batch_data,batch_tag)
            train_loss.backward()
            opt.step()
            opt.zero_grad()
            # print(f"train_loss:{train_loss:.3f}")
        model.eval()
        all_pre = []
        all_tag = []
        for dev_batch_data,dev_batch_tag,batch_len in dev_dataloader:
            pre_tag = model.test(dev_batch_data,batch_len)
            all_pre.extend(pre_tag.detach().cpu().numpy().tolist())
            all_tag.extend(dev_batch_tag[:,:-1].detach().cpu().numpy().reshape(-1).tolist())
        print('all_tag:',len(all_tag))
        print('all_pre:',len(all_pre))
        score = f1_score(all_tag,all_pre,average="micro")
        print(f"{e},f1_score:{score:.3f},train_loss:{train_loss:.3f}")
    torch.save(model, 'myModel.pkl')
    test()
