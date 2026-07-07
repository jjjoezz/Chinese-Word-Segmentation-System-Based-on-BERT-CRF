import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BertCWS(nn.Module):
    def __init__(self, tag2id, model_name='bert-base-chinese'):
        super(BertCWS, self).__init__()

        # 1. 加载预训练的 BERT
        self.bert = BertModel.from_pretrained(model_name)
        # 添加 Dropout 防过拟合
        self.dropout = nn.Dropout(0.3)

        # 2. 线性映射层：768维 -> 标签数量
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, len(tag2id))

        # 3. CRF 条件随机场
        self.crf = CRF(len(tag2id), batch_first=True)

    # ✅ forward 必须在类中，而不是 __init__ 里
    def forward(self, input_ids, attention_mask, label):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask.long())
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        emissions = self.hidden2tag(sequence_output)
        loss = -self.crf(emissions, label, mask=attention_mask.bool(),
                         reduction='mean')
        return loss

    # ✅ infer 同理
    def infer(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask.long())
        sequence_output = outputs.last_hidden_state
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, mask=attention_mask.bool())