import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pickle
import logging
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # BERT 必须用 AdamW
from model import BertCWS
from dataloader import BertDataset


def get_param():
    parser = argparse.ArgumentParser()
    # BERT 的黄金学习率
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--max_epoch', type=int, default=10)
    # Batch Size 控制在 16 防止爆显存
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()


def set_logger():
    os.makedirs('save', exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='save/log.txt',
        filemode='w',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


def entity_split(x, y, id2tag, entities, cur):
    start, end = -1, -1
    for j in range(len(x)):
        tag = id2tag[y[j]]  # ✅ 先转换为字符串标签
        if tag == 'B':
            start = cur + j
        elif tag == 'M' and start != -1:
            continue
        elif tag == 'E' and start != -1:
            end = cur + j
            entities.add((start, end))
            start, end = -1, -1
        elif tag == 'S':
            entities.add((cur + j, cur + j))
            start, end = -1, -1
        else:
            start, end = -1, -1


def main(args):
    use_cuda = torch.cuda.is_available()

    print("正在加载数据...")
    with open('data/bert_data.pkl', 'rb') as inp:
        tag2id, id2tag, x_train, y_train, x_test, y_test = pickle.load(inp)

    print("正在加载 BERT 模型...")
    model = BertCWS(tag2id)
    if use_cuda:
        model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    train_data = DataLoader(
        BertDataset(x_train, y_train),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=BertDataset.collate_fn
    )

    test_data = DataLoader(
        BertDataset(x_test[:1000], y_test[:1000]),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=BertDataset.collate_fn
    )

    print("🚀 开始狂飙！")
    for epoch in range(args.max_epoch):
        model.train()
        step, log = 0, []

        # ✅ 训练循环
        for input_ids, label, mask, length in train_data:
            if use_cuda:
                input_ids, label, mask = input_ids.cuda(), label.cuda(), mask.cuda()

            loss = model(input_ids, mask, label)
            log.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % 50 == 0:
                logging.info('epoch %d-step %d loss: %f' % (epoch, step,
                                                            sum(log) / len(log)))
                log = []

        # ✅ 测试环节（在训练循环外部）
        entity_predict, entity_label = set(), set()
        with torch.no_grad():
            model.eval()
            cur = 0
            for input_ids, label, mask, length in test_data:
                if use_cuda:
                    input_ids, label, mask = input_ids.cuda(), label.cuda(), mask.cuda()

                predicts = model.infer(input_ids, mask)

                for i in range(len(length)):
                    real_len = length[i] - 2
                    if real_len <= 0:
                        continue

                    pred_tags = predicts[i][1:real_len + 1]
                    true_tags = label[i][1:real_len + 1].cpu().tolist()
                    dummy_x = [0] * real_len

                    entity_split(dummy_x, pred_tags, id2tag, entity_predict, cur)
                    entity_split(dummy_x, true_tags, id2tag, entity_label, cur)
                    cur += real_len

        # ✅ 计算指标（在 with torch.no_grad() 外部）
        right_predict = [i for i in entity_predict if i in entity_label]
        if len(right_predict) != 0 and len(entity_predict) != 0 and len(entity_label) != 0:
            p = float(len(right_predict)) / len(entity_predict)
            r = float(len(right_predict)) / len(entity_label)
            f1 = (2 * p * r) / (p + r)
            logging.info(f"====== Epoch {epoch} 战报 ======")
            logging.info("precision: %f" % p)
            logging.info("recall: %f" % r)
            logging.info("fscore: %f" % f1)
            logging.info("================================")

        # ✅ 保存模型（在每个epoch结束后）
        path_name = f"./save/bert_model_epoch{epoch}.pkl"
        torch.save(model, path_name)


if __name__ == '__main__':
    set_logger()
    main(get_param())