import torch
import pickle
from transformers import BertTokenizer

if __name__ == '__main__':
    print("正在加载 Tokenizer...")
    #tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-chinese',
        local_files_only=True  # ✅ 只使用本地缓存
    )
    best_epoch = 1
    print(f"正在加载训练好的 BERT 模型 (Epoch {best_epoch})...")
    model = torch.load(f'save/bert_model_epoch{best_epoch}.pkl',
                       map_location='cpu', weights_only=False)
    model.eval()

    with open('data/bert_data.pkl', 'rb') as inp:
        tag2id, id2tag, _, _, _, _ = pickle.load(inp)

    output = open('cws_result.txt', 'w', encoding='utf-8')

    print("正在对 test_data.txt 进行疯狂分词...")
    with open('data/test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                print(file=output)
                continue

            chars = list(line)

            # 超长句无损分块处理 (Chunking)
            MAX_CHUNK_SIZE = 510  # 留2个位置给 [CLS] 和 [SEP]
            final_predict = []

            for i in range(0, len(chars), MAX_CHUNK_SIZE):
                chunk_chars = chars[i: i + MAX_CHUNK_SIZE]
                tokens = ['[CLS]'] + [tokenizer.tokenize(c)[0] if
                                      len(tokenizer.tokenize(c)) > 0 else '[UNK]' for c in chunk_chars] + ['[SEP]']
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                x = torch.tensor([input_ids], dtype=torch.long)
                mask = torch.ones_like(x, dtype=torch.bool)

                with torch.no_grad():
                    predict = model.infer(x, mask)[0]

                    # 剥离预测结果头尾的 [CLS] 和 [SEP]，然后无缝拼接
                    final_predict.extend(predict[1:-1])

            # 最终输出到文件
            for i in range(len(chars)):
                print(chars[i], end='', file=output)
                if id2tag[final_predict[i]] in ['E', 'S']:
                    print(' ', end='', file=output)
            print(file=output)

    # ✅ 正确：在所有行处理完毕后关闭文件
    output.close()
    print("文件已生成！(cws_result.txt)")