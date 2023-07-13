# -*- coding:utf-8 -*-

import gc
import glob
import json
import os
import torch
from os.path import join as pjoin

from collections import Counter
from rouge import Rouge
from others.logging import logger
# from others.tokenization import BertTokenizer # 中文bert
from transformers import BertTokenizer # 英文bert
from others.vocab_wrapper import VocabWrapper


def greedy_selection(doc, summ, summary_size):
    # 实现了一个贪心算法，用于从一组文档中选择出一些句子组成摘要
    doc_sents = list(map(lambda x: x["original_txt"], doc))
    # print(doc_sents)
    # assert 0
    max_rouge = 0.0

    rouge = Rouge()
    selected = []
    while True:
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(doc_sents)):
            if (i in selected):
                continue
            c = selected + [i]
            temp_txt = " ".join([doc_sents[j] for j in c])
            if len(temp_txt.split()) > summary_size:
                continue
            rouge_score = rouge.get_scores(temp_txt, summ)
            # print(rouge_score)
            # assert 0
            rouge_1 = rouge_score[0]["rouge-1"]["r"]
            rouge_2 = rouge_score[0]["rouge-2"]["r"]
            rouge_l = rouge_score[0]["rouge-l"]["r"]
            rouge_score = rouge_1 + rouge_2 + rouge_l
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return selected


class BertData():
    def __init__(self, args):
        self.args = args
        # self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir) # 中文bert
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased') # 英文bert

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.tgt_bos = '[unused1]'
        self.tgt_eos = '[unused2]'
        # self.role_1 = '[unused3]'
        # self.role_2 = '[unused4]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.unk_vid = self.tokenizer.vocab[self.unk_token]

    def preprocess_src(self, content, info=None):
        if_exceed_length = False

        # print(info)
        # print(content)
        # assert 0
        # if not (info == "客服" or info == '客户'):
        #     return None
        if len(content) < self.args.min_src_ntokens_per_sent:
            return None
        if len(content) > self.args.max_src_ntokens_per_sent:
            if_exceed_length = True

        original_txt = ' '.join(content)

        # print(original_txt) # 一个session中的所有content拼起来
       # assert 0

        if self.args.truncated: # 正常运行会进入该分支
            content = content[:self.args.max_src_ntokens_per_sent] # 从设置的最大词数截断
            # print(content)
            # assert 0
        # content_text = ' '.join(content).lower() # 全部转小写，如果用bert-base-cased不用转，因为chinese_bert中vocab中的英文都是小写的，所以这里有一个转小写操作。
        # print(type(content_text)) # <class 'str'>
        # assert 0
        # content_subtokens = self.tokenizer.tokenize(content_text) # 这里是中文分词的过程，我的英文数据集在转换格式部分已经用bert做过分词
        content_subtokens = content
        # print(content_subtokens)
        # assert 0

        # [CLS] + T0 + T1 + ... + Tn
        # if info == '客服':
        #     src_subtokens = [self.cls_token, self.role_1] + content_subtokens
        # else:
        #     src_subtokens = [self.cls_token, self.role_2] + content_subtokens

        src_subtokens = [self.cls_token] + content_subtokens
        # print(src_subtokens)
        # assert 0
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        # print(src_subtoken_idxs)
        # assert 0
        segments_ids = len(src_subtoken_idxs) * [0]
        # print(segments_ids) # 全是0
        # assert 0

        return src_subtoken_idxs, segments_ids, original_txt, src_subtokens, if_exceed_length

    def preprocess_summary(self, content):

        original_txt = ' '.join(content)

        # content_text = ' '.join(content).lower()
        # content_subtokens = self.tokenizer.tokenize(content_text)

        content_subtokens = content

        content_subtokens = [self.tgt_bos] + content_subtokens + [self.tgt_eos]
        subtoken_idxs = self.tokenizer.convert_tokens_to_ids(content_subtokens)

        # print(subtoken_idxs, original_txt, content_subtokens) # original_txt？保留原始文本内容，目前先用分词之后的结果，后面看一下original_txt用在那里，如果只是输出不参与训练则没事，参与训练则要看是否适合
        # assert 0
        return subtoken_idxs, original_txt, content_subtokens

    def integrate_dialogue(self, dialogue):
        src_tokens = [self.cls_token]
        segments_ids = [0]
        segment_id = 0
        for sent in dialogue:
            tokens = sent["src_tokens"][1:] + [self.sep_token]
            src_tokens.extend(tokens)
            segments_ids.extend([segment_id] * len(tokens))
            segment_id = 1 - segment_id # 每次取反，这样00001111100000，就能把句子分开
        src_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
        # print({"src_id": src_ids, "segs": segments_ids})
        # assert 0
        return {"src_id": src_ids, "segs": segments_ids}


def topic_info_generate(dialogue, file_counter):
    all_counter = Counter() # 记录每个单词（token_id）出现的次数，降序
    customer_counter = Counter()
    agent_counter = Counter()

    for sent in dialogue:
        # print(sent) # 句子的tokenize过程中存储的相关信息
        # assert 0
        role = sent["role"]
        token_ids = sent["tokenized_id"]
        all_counter.update(token_ids)
        # 下面这段role的if else分支没用了，因为role的信息我设置都为空
        # if role == "客服":
        #     agent_counter.update(token_ids)
        # else:
        #     customer_counter.update(token_ids)
    file_counter['all'].update(all_counter.keys())
    file_counter['customer'].update(customer_counter.keys())
    file_counter['agent'].update(agent_counter.keys())
    file_counter['num'] += 1
    return {"all": all_counter, "customer": customer_counter, "agent": agent_counter}


def topic_summ_info_generate(dialogue, ex_labels):
    all_counter = Counter()
    customer_counter = Counter()
    agent_counter = Counter()

    for i, sent in enumerate(dialogue):
        if i in ex_labels:
            role = sent["role"]
            token_ids = sent["tokenized_id"]
            all_counter.update(token_ids)
            # role输入为空，则全部为全局，没有客户/客服
            # if role == "客服":
            #     agent_counter.update(token_ids)
            # else:
            #     customer_counter.update(token_ids)
    return {"all": all_counter, "customer": customer_counter, "agent": agent_counter}


def topic_summ_info_generate_split(dialogue, verdict_ex_labels, pros_ex_labels, cons_ex_labels):
    all_counter = Counter()
    customer_counter = Counter()
    agent_counter = Counter()

    for i, sent in enumerate(dialogue):
        if i in verdict_ex_labels: # 整体
            token_ids = sent["tokenized_id"]
            all_counter.update(token_ids)
        if i in pros_ex_labels: # 优点
            token_ids = sent["tokenized_id"]
            customer_counter.update(token_ids)
        if i in cons_ex_labels: # 缺点
            token_ids = sent["tokenized_id"]
            agent_counter.update(token_ids)
    return {"all": all_counter, "customer": customer_counter, "agent": agent_counter}


def format_to_bert(args, corpus_type=None):

    a_lst = []
    file_counter = {"all": Counter(), "customer": Counter(), "agent": Counter(), "num": 0, "voc_size": 0}
    if corpus_type is not None:
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, file_counter, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
    else:
        for json_f in glob.glob(pjoin(args.raw_path, '*.json')):
            real_name = json_f.split('/')[-1] # linux
            # real_name = json_f.split('\\')[-1] # windows
            # print(real_name)
            corpus_type = real_name.split('.')[1]
            # print(corpus_type) # dev/test/train
            # assert 0
            a_lst.append((corpus_type, json_f, args, file_counter, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))

    total_statistic = {
        "instances": 0,
        "total_turns": 0.,
        "processed_turns": 0.,
        "max_turns": -1,
        "turns_num": [0] * 11,
        "exceed_length_num": 0,
        "exceed_turns_num": 0,
        "total_src_length": 0.,
        "src_sent_length_num": [0] * 11,
        "src_token_length_num": [0] * 11,
        "total_tgt_length": 0
    }
    for d in a_lst:
        statistic = _format_to_bert(d)
        if statistic is None:
            continue
        total_statistic["instances"] += statistic["instances"]
        total_statistic["total_turns"] += statistic["total_turns"]
        total_statistic["processed_turns"] += statistic["processed_turns"]
        total_statistic["max_turns"] = max(total_statistic["max_turns"], statistic["max_turns"])
        total_statistic["exceed_length_num"] += statistic["exceed_length_num"]
        total_statistic["exceed_turns_num"] += statistic["exceed_turns_num"]
        total_statistic["total_src_length"] += statistic["total_src_length"]
        total_statistic["total_tgt_length"] += statistic["total_tgt_length"]
        for idx in range(len(total_statistic["turns_num"])):
            total_statistic["turns_num"][idx] += statistic["turns_num"][idx]
        for idx in range(len(total_statistic["src_sent_length_num"])):
            total_statistic["src_sent_length_num"][idx] += statistic["src_sent_length_num"][idx]
        for idx in range(len(total_statistic["src_token_length_num"])):
            total_statistic["src_token_length_num"][idx] += statistic["src_token_length_num"][idx]

    # save file counter
    save_file = pjoin(args.save_path, 'idf_info.pt')
    logger.info('Saving file counter to %s' % save_file)
    torch.save(file_counter, save_file)
    print("***************************")
    print(file_counter) # 这里如果在存在alli.dev.0.bert.pt等文件情况下运行，会忽略掉前面的步骤，但是这里会把file_counter里面存的内容清空为初试状态。
    print("***************************")

    if total_statistic["instances"] > 0:
        logger.info("Total examples: %d" % total_statistic["instances"])
        logger.info("Average sentence number per dialogue: %f" % (total_statistic["total_turns"] / total_statistic["instances"]))
        logger.info("Processed average sentence number per dialogue: %f" % (total_statistic["processed_turns"] / total_statistic["instances"]))
        logger.info("Total sentences: %d" % total_statistic["total_turns"])
        logger.info("Processed sentences: %d" % total_statistic["processed_turns"])
        logger.info("Exceeded max sentence number dialogues: %d" % total_statistic["exceed_turns_num"])
        logger.info("Max dialogue sentences: %d" % total_statistic["max_turns"])
        for idx, num in enumerate(total_statistic["turns_num"]):
            logger.info("Dialogue sentences %d ~ %d: %d, %.2f%%" % (idx * 20, (idx+1) * 20, num, (num / total_statistic["instances"])))
        logger.info("Exceed length sentences number: %d" % total_statistic["exceed_length_num"])
        logger.info("Average src sentence length: %f" % (total_statistic["total_src_length"] / total_statistic["total_turns"]))
        for idx, num in enumerate(total_statistic["src_sent_length_num"]):
            logger.info("Sent length %d ~ %d: %d, %.2f%%" % (idx * 10, (idx+1) * 10, num, (num / total_statistic["total_turns"])))
        logger.info("Average src token length: %f" % (total_statistic["total_src_length"] / total_statistic["instances"]))
        for idx, num in enumerate(total_statistic["src_token_length_num"]):
            logger.info("token num %d ~ %d: %d, %.2f%%" % (idx * 300, (idx+1) * 300, num, (num / total_statistic["instances"])))
        logger.info("Average tgt length: %f" % (total_statistic["total_tgt_length"] / total_statistic["instances"]))


def _format_to_bert(params):
    _, json_file, args, file_counter, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    if args.tokenize: # 正常情况下进这个分支
        # print("进入第一个tokenize if分支")
        voc_wrapper = VocabWrapper(args.emb_mode)
        voc_wrapper.load_emb(args.emb_path)
        file_counter['voc_size'] = voc_wrapper.voc_size()

    datasets = [] # 存储数据集一个json文件预处理后的数据
    exceed_length_num = 0
    exceed_turns_num = 0
    total_src_length = 0.
    total_tgt_length = 0.
    src_length_sent_num = [0] * 11
    src_length_token_num = [0] * 11
    max_turns = 0
    turns_num = [0] * 11
    dialogue_turns = 0.
    processed_turns = 0.

    count = 0

    for dialogue in jobs:
        dialogue_b_data = []
        dialogue_token_num = 0
        for index, sent in enumerate(dialogue['session']):
            content = sent['content']
            role = sent['type']
            b_data = bert.preprocess_src(content, role) # 注意content太长会截断
            # b_data(评论句子使用bert-base-cased的token_id, 句子标识id, 原评论文本，[CLS]评论句子分词，是否发生截断)
            if (b_data is None):
                continue
            src_subtoken_idxs, segments_ids, original_txt, src_subtokens, exceed_length = b_data
            b_data_dict = {"index": index, "src_id": src_subtoken_idxs,
                           "segs": segments_ids, "original_txt": original_txt,
                           "src_tokens": src_subtokens, "role": role} # b_data的字典形式
            if args.tokenize: # 正常情况进这个分支
                # print("进入第二个tokenize if分支")
                ids = map(lambda x: voc_wrapper.w2i(x), sent['word']) # word2vec训练好的词向量，这里调用加载了word2vec的vocab_wrapper这个类的w2i，就是把word2vec训练出来词向量，根据word找到index索引
                tokenized_id = [x for x in ids if x is not None]
                # print(tokenized_id)
                # assert 0
                b_data_dict["tokenized_id"] = tokenized_id # word2vec的token_id
            else: # 正常情况不进这个分支
                # b_data_dict["tokenized_id"] = src_subtoken_idxs[2:] # 这里从2开始是因为之前添加了角色信息符号role_1/role_2
                b_data_dict["tokenized_id"] = src_subtoken_idxs[1:] # 这里从1开始是取消了之前添加了角色信息符号role_1/role_2
            src_length_sent_num[min(len(src_subtoken_idxs) // 10, 10)] += 1
            dialogue_token_num += len(src_subtoken_idxs)
            total_src_length += len(src_subtoken_idxs)
            dialogue_b_data.append(b_data_dict) # 存储每条评论的b_data_dict  
            if exceed_length:
                exceed_length_num += 1
            if len(dialogue_b_data) >= args.max_turns: # max_turns=100
                exceed_turns_num += 1
                if args.truncated:
                    # print("进入truncated if分支")
                    break
        dialogue_example = {"session": dialogue_b_data} # 一个产品的相关所有评论的b_data_dict
        dialogue_integrated = bert.integrate_dialogue(dialogue_b_data) # 包括src_id(bert-base-cased分词)，seg_id(1,1,1,...0,0,0)分句
        topic_info = topic_info_generate(dialogue_b_data, file_counter) #  # 存储评论中all的BOW、customer、agent为空
        dialogue_example["dialogue"] = dialogue_integrated
        dialogue_example["topic_info"] = topic_info
        # dialogue_example{"session":一个产品的相关所有评论的b_data_dict, "dialogue":包括src_id(bert-base-cased分词)，seg_id(1,1,1,...0,0,0)分句, "topic_info":评论中all的BOW、customer、agent为空}

        # test & dev data process
        if "summary" in dialogue.keys():
            content = dialogue["summary"]

            if '[unused13]' in content:
                if content.count('[unused13]') == 1:
                    index = content.index('[unused13]')
                    verdict = content[:index]
                    pros = content[index+1:]
                    cons = ['']
                else:
                    index1 = content.index('[unused13]')
                    index2 = content.index('[unused13]', index1+1)
                    verdict = content[:index1]
                    pros = content[index1+1:index2]
                    cons = content[index2+1:]
            else:
                verdict = content
                pros = ['']
                cons = ['']
            verdict_original_txt = ' '.join(verdict)
            verdict_original_txt = verdict_original_txt.replace(" ##", "")
            pros_original_txt = ' '.join(pros)
            pros_original_txt = pros_original_txt.replace(" ##", "")
            cons_original_txt = ' '.join(cons)
            cons_original_txt = cons_original_txt.replace(" ##", "")

            summ_b_data = bert.preprocess_summary(content)
            # print(summ_b_data)
            # assert 0
            subtoken_idxs, original_txt, content_subtokens = summ_b_data
            total_tgt_length += len(subtoken_idxs)
            b_data_dict = {"id": subtoken_idxs,
                           "original_txt": original_txt,
                           "content_tokens": content_subtokens}
            if args.add_ex_label:
                # print("进入add_ex_label if分支") # 正常运行会进入该分支
                ex_labels = greedy_selection(dialogue_b_data, original_txt, args.ex_max_token_num)
                # verdict_ex_labels = greedy_selection(dialogue_b_data, verdict_original_txt, args.ex_max_token_num) # 需统计整体部分的长度，也可以不统计？
                # pros_ex_labels = greedy_selection(dialogue_b_data, pros_original_txt, args.ex_max_token_num) # 需统计优点部分的长度
                # cons_ex_labels = greedy_selection(dialogue_b_data, cons_original_txt, args.ex_max_token_num) # 需统计缺点部分的长度
                verdict_ex_labels = greedy_selection(dialogue_b_data, verdict_original_txt, 150) # 需统计整体部分的长度，也可以不统计？
                pros_ex_labels = greedy_selection(dialogue_b_data, pros_original_txt, 150) # 需统计优点部分的长度
                cons_ex_labels = greedy_selection(dialogue_b_data, cons_original_txt, 150) # 需统计缺点部分的长度
                # print(ex_labels) # [56, 3, 33, 91, 8]
                # print(verdict_ex_labels) # [46, 61, 8]
                # print(pros_ex_labels) # [3, 91]
                # print(cons_ex_labels) # [2]
                # topic_summ_info = topic_summ_info_generate(dialogue_b_data, ex_labels)
                all_labels = list(set(ex_labels + verdict_ex_labels + pros_ex_labels + cons_ex_labels)) # 这样可以让抽取的评论（抽取器的标签）更加多
                # print(all_labels) # [33, 2, 3, 8, 46, 56, 91, 61]
                topic_summ_info = topic_summ_info_generate_split(dialogue_b_data, verdict_ex_labels, pros_ex_labels, cons_ex_labels)
                # print(topic_summ_info)
                # {'all': Counter({0: 17, 5: 7, 1: 6, 4: 6, 3: 6, 10: 4, 14: 4, 12: 4, 16: 3, 249: 3, 29: 3, 11: 3, 77: 2, 124: 2, 56: 2, 6: 2, 45: 2, 25: 2, 8: 2, 139: 2, 34: 2, 13: 2, 52: 2, 9: 2, 49: 2, 115: 2, 2: 2, 250: 1, 142: 1, 105: 1, 83: 1, 70: 1, 84: 1, 259: 1, 71: 1, 211: 1, 280: 1, 22: 1, 41: 1, 61: 1, 111: 1, 50: 1, 51: 1, 108: 1, 44: 1, 114: 1, 27: 1, 170: 1, 242: 1, 197: 1, 79: 1, 272: 1, 17: 1, 88: 1, 18: 1, 133: 1, 195: 1, 220: 1, 24: 1, 39: 1, 54: 1, 172: 1, 281: 1, 121: 1, 46: 1, 107: 1, 188: 1}), 'customer': Counter({0: 13, 2: 10, 6: 7, 13: 6, 1: 6, 8: 5, 10: 4, 7: 4, 4: 3, 25: 3, 3: 3, 69: 3, 16: 3, 27: 2, 67: 2, 9: 2, 33: 2, 45: 2, 28: 2, 38: 2, 5: 2, 58: 2, 14: 2, 24: 1, 81: 1, 115: 1, 102: 1, 132: 1, 51: 1, 207: 1, 270: 1, 192: 1, 12: 1, 73: 1, 145: 1, 50: 1, 155: 1, 77: 1, 82: 1, 298: 1, 11: 1, 57: 1, 29: 1, 159: 1, 59: 1, 44: 1, 260: 1, 127: 1, 230: 1, 269: 1, 121: 1, 84: 1, 75: 1, 26: 1, 74: 1, 37: 1, 122: 1, 110: 1, 181: 1, 15: 1, 65: 1, 96: 1, 19: 1, 42: 1, 183: 1, 70: 1, 209: 1, 130: 1, 61: 1, 111: 1, 193: 1, 281: 1, 30: 1, 92: 1, 258: 1}), 'agent': Counter({8: 6, 1: 6, 0: 5, 172: 4, 5: 4, 2: 3, 12: 3, 14: 3, 6: 3, 25: 3, 9: 2, 10: 2, 181: 2, 259: 2, 24: 1, 39: 1, 19: 1, 88: 1, 258: 1, 46: 1, 37: 1, 31: 1, 296: 1, 216: 1, 87: 1, 23: 1, 7: 1, 30: 1, 111: 1, 217: 1, 223: 1, 17: 1, 248: 1, 297: 1, 129: 1, 280: 1, 45: 1, 51: 1, 115: 1, 110: 1, 18: 1, 245: 1, 84: 1, 132: 1})}
                b_data_dict["ex_labels"] = all_labels # 最终抽取器的标签用三个部分抽取和总抽取，共四个抽取结果的并集来表示
                b_data_dict["topic_summ_info"] = topic_summ_info # 分别对应verdict,pros,cons的BOW（词袋）
            dialogue_example["summary"] = b_data_dict
            # print(dialogue_example) # 存储好评论和摘要的预处理的相关信息
            # assert 0

        if len(dialogue_b_data) >= args.min_turns:
            datasets.append(dialogue_example)
            turns_num[min(len(dialogue_b_data) // 20, 10)] += 1
            src_length_token_num[min(dialogue_token_num // 300, 10)] += 1
            max_turns = max(max_turns, len(dialogue_b_data))
            dialogue_turns += len(dialogue['session'])
            processed_turns += len(dialogue_b_data)

            count += 1
            # print(count)
            if count % 50 == 0:
                print(count)
            # assert 0

    statistic = {
        "instances": len(datasets),
        "total_turns": dialogue_turns,
        "processed_turns": processed_turns,
        "max_turns": max_turns,
        "turns_num": turns_num,
        "exceed_length_num": exceed_length_num,
        "exceed_turns_num": exceed_turns_num,
        "total_src_length": total_src_length,
        "src_sent_length_num": src_length_sent_num,
        "src_token_length_num": src_length_token_num,
        "total_tgt_length": total_tgt_length
    }

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
    return statistic
