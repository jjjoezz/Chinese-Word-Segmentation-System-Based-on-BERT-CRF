import os
# 设置国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pickle
import os
from transformers import BertTokenizer

def convert_to_bert():
    print("正在加载 BERT Tokenizer...")
    # 尝试从本地加载，如果本地没有，则从镜像下载
    local_model_path = './bert-base-chinese'
    if os.path.exists(local_model_path):
        print(f"从本地路径 {local_model_path} 加载模型...")
        tokenizer = BertTokenizer.from_pretrained(local_model_path)
    else:
        print("从 Hugging Face 镜像下载模型...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    print("正在读取旧的 datasave.pkl...")
    with open('datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    def process_data(x_data, y_data):
        bert_x, bert_y = [], []
        for i in range(len(x_data)):
            # 安全兼容 id2word 是 list 或 dict 的情况
            chars = []
            for idx in x_data[i]:
                if isinstance(id2word, list):
                    # 如果是列表，通过索引访问，并防止索引越界
                    chars.append(id2word[idx] if idx < len(id2word) else '[UNK]')
                elif isinstance(id2word, dict):
                    # 如果是字典，使用 get 方法
                    chars.append(id2word.get(idx, '[UNK]'))
                else:
                    chars.append('[UNK]')

            tokens = ['[CLS]']
            for c in chars:
                tok = tokenizer.tokenize(c)
                tokens.append(tok[0] if len(tok) > 0 else '[UNK]')
            tokens.append('[SEP]')

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # 安全获取标签 'S' 的 ID 用于填充
            if isinstance(tag2id, dict):
                s_tag_id = tag2id.get('S', 0)
            else:
                s_tag_id = 0

            labels = [s_tag_id] + y_data[i] + [s_tag_id]

            bert_x.append(input_ids)
            bert_y.append(labels)
        return bert_x, bert_y

    print("正在将训练集转换为 BERT 格式...")
    bert_x_train, bert_y_train = process_data(x_train, y_train)
    print("正在将测试集转换为 BERT 格式...")
    bert_x_test, bert_y_test = process_data(x_test, y_test)

    os.makedirs('data', exist_ok=True)
    with open('data/bert_data.pkl', 'wb') as f:
        pickle.dump((tag2id, id2tag, bert_x_train, bert_y_train, bert_x_test,
                     bert_y_test), f)
    print("大功告成！全新的 BERT 数据集已保存至 data/bert_data.pkl")

if __name__ == '__main__':
    convert_to_bert()