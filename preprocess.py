# encoding=utf-8

import argparse
from others.logging import init_logger
from prepro import data_builder as data_builder


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)

    parser.add_argument("-type", default='train', type=str)
    parser.add_argument("-raw_path", default='json_data')
    parser.add_argument("-save_path", default='bert_data')
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument("-idlize", nargs='?', const=True, default=False)

    # parser.add_argument("-bert_dir", default='bert/chinese_bert') # 换成英文bert
    parser.add_argument('-min_src_ntokens', default=1, type=int)
    parser.add_argument('-max_src_ntokens', default=3000, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=10, type=int)  # 每条评论(content)的最小长度10
    # parser.add_argument('-max_src_ntokens_per_sent', default=700, type=int) # 每条评论(content)的最大长度700
    parser.add_argument('-max_src_ntokens_per_sent', default=150, type=int) # 每条评论(content)的最大长度150
    parser.add_argument('-min_tgt_ntokens', default=10, type=int)    # 生成摘要（分词）最短长度为10
    # parser.add_argument('-max_tgt_ntokens', default=700, type=int)  # 生成摘要（分词）最长长度为700
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)  # 生成摘要（分词）最长长度为700
    parser.add_argument('-min_turns', default=1, type=int)      # 最小1条评论
    parser.add_argument('-max_turns', default=100, type=int)    # 最多100条评论
    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-tokenize", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-emb_mode", default="word2vec", type=str, choices=["glove", "word2vec"])
    parser.add_argument("-emb_path", default="pretrain_emb/word2vec", type=str)
    # parser.add_argument("-ex_max_token_num", default=700, type=int) # 贪心抽取出来的最大摘要（分词）长度
    parser.add_argument("-ex_max_token_num", default=500, type=int) # 贪心抽取出来的最大摘要（分词）长度
    parser.add_argument("-truncated", nargs='?', const=True, default=False)
    parser.add_argument("-add_ex_label", nargs='?', const=True, default=False)

    parser.add_argument('-log_file', default='logs/preprocess.log')
    parser.add_argument('-dataset', default='')

    args = parser.parse_args()
    if args.type not in ["train", "dev", "test"]:
        print("Invalid data type! Data type should be 'train', 'dev', or 'test'.")
        exit(0)
    init_logger(args.log_file)
    data_builder.format_to_bert(args)