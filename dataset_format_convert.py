# 将AMASUM数据集的格式转换为对话摘要数据集的格式
# amasum数据集                         对话摘要数据集
# customer_reviews               ->       session
# website_summaries              ->       summary
# verdict+分割词+pros+分割词+cons  ->      summary的value
# customer_reviews中的text        ->      content的value
# customer_reviews中的text        ->      word的value
# ""                             ->      type的value

import argparse # 命令行选项/参数/子命令 解释
import os       # 路径
import json     # 处理json
import glob     # 查找文件
from os.path import join as pjoin                       # 安装Hugging Face：pip install transformers==3.4.0
from transformers import BertTokenizer, BertModel       # Hugging Face: https://huggingface.co/docs/tokenizers/index

def format_converter(args, summ_length_list):
    print("Start convert the format...") # test/train/vaild
    data_dir = os.path.abspath(args.data_path)
    print("Preparing to process %s ..." % data_dir) # 正在处理的数据集的路径
    raw_files = glob.glob(pjoin(data_dir, '*.json')) # 查找data_dir路径下的所有json文件并存入raw_files中

    re_su_pairs = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased') # 英文bert
    for s in raw_files: # 遍历所有的test/train/vaild下的json文件
        exs = json.load(open(s))
        print("Processing File "+ s)

        # reviews
        # 取出customer_reviews列表中每一个的text
        reviews = []
        ex_num= 0
        for i in exs['customer_reviews']:
            # print(i['text'])
            # assert 0
            review = tokenizer.tokenize(i['text'])
            reviews.append(review)
            # review_length_list.append(len(review))
            # print(review)
            # print(len(review))
            # assert 0
            ex_num += 1
        # print(reviews)
        # print(type(reviews)) # <class 'list'>
        # print(reviews[0])
        # print(type(reviews[0])) # <class 'list'>
        # print(ex_num)
        # assert 0

        # break
        # summaries
        # 取出website_summaries的verdict/pros/cons，并添加分割词，因为这里需要verdict/pros/cons需要分开评估
        summaries = []
        summary = exs['website_summaries'][0]
        summaries.extend(tokenizer.tokenize(summary['verdict'])) # verdict
        summaries.extend(['[unused13]']) # 分割词
        for i in summary['pros']: summaries.extend(tokenizer.tokenize(i)) # pros
        summaries.extend(['[unused13]']) # 分割词
        for i in summary['cons']: summaries.extend(tokenizer.tokenize(i)) # cons
        summ_length_list.append(len(summaries)) # 存储分词以后总摘要长度
        # print(len(summaries))
        # print(summaries)
        # print(type(summaries)) # <class 'list'>
        # assert 0

        # 将取出的评论和摘要组成新的格式（对话摘要数据集格式）
        session = []
        for content in reviews:
            session.append({"content": content, "word": content, "type": ''})
        re_su_pair = {"session": session, "summary": summaries}
        re_su_pairs.append(re_su_pair)
        # print(re_su_pair)
        # assert 0

    # print(re_su_pairs)
    # print(len(re_su_pairs))
    # assert 0
    # if (args.mode == "test"):
    #     with open ('json_data/ali.test.0.json','w+') as f: # w+会覆盖 a+不覆盖
    #         json.dump(re_su_pairs,f)
    # elif (args.mode == "train"):
    #     num_files = 9  # 存储train中数据的文件数量
    #     num_re_su_pair = len(re_su_pairs)  # 数据数量
    #     max_num_single_file = 3160  # 每个文件的数据数量不超过3160个re_su_pair
    #     # max_num_single_file = 3  # 测试3个
    #     for i in range(num_files):
    #         start_idx = i * max_num_single_file
    #         end_idx = min((i + 1) * max_num_single_file, num_re_su_pair)
    #         file_path = f'json_data/ali.train.{i}.json'  # 以f开头表示在字符串内支持大括号内的python表达式（Python3.6新增）
    #         with open(file_path, 'w+') as f:
    #             json.dump(re_su_pairs[start_idx:end_idx], f)
    #         if end_idx >= num_re_su_pair:
    #             break
    # elif (args.mode == "valid"):
    #     with open ('json_data/ali.dev.0.json','w+') as f:
    #         json.dump(re_su_pairs,f)
    # print("The json file has been written!")
    # assert 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", default='amasum_raw_dataset/test', type=str) # 为了先调试后续模型跑通，文件个数缩减 3166->3
    parser.add_argument("-mode", default="test", type=str, choices=["test", "train", "valid"])
    # parser.add_argument("-data_path", default='amasum_raw_dataset/train', type=str) # 文件个数缩减 25203->8
    # parser.add_argument("-mode", default="train", type=str, choices=["test", "train", "valid"])
    # parser.add_argument("-data_path", default='amasum_raw_dataset/valid', type=str) # 文件个数缩减 3114->3
    # parser.add_argument("-mode", default="vaild", type=str, choices=["test", "train", "valid"])

    args = parser.parse_args()
    # print(args)

    # review_length_list = []  # 存储每条评论分词以后长度
    # format_converter(args, review_length_list)
    # print(max(review_length_list))
    # with open('args_related/re_len_list.json', 'w+') as f:
    #      json.dump(review_length_list,f)
    # print("The args json file has been written!")

    summ_length_list = []  # 存储每个黄金摘要分词以后长度
    format_converter(args, summ_length_list)
    with open('args_related/summ_len_list.json', 'w+') as f:
        json.dump(summ_length_list, f)
    print("The args json file has been written!")