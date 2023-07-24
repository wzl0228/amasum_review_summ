#!/usr/bin/env python
# -*-coding:utf8-*-
""" Translator Class and builder """
from __future__ import print_function
import codecs
import torch

from tensorboardX import SummaryWriter
from others.utils import rouge_results_to_str, test_bleu, test_length, test_f1
from translate.beam import GNMTGlobalScorer
from rouge import Rouge, FilesRouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def build_predictor(args, tokenizer, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')

    translator = Translator(args, model, tokenizer,
                            global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 tokenizer,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        # self.generator = self.model.generator # 对应修改为三个，没用到？
        self.verdict_generator = self.model.verdict_generator
        self.pros_generator = self.model.pros_generator
        self.cons_generator = self.model.cons_generator
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.start_token = self.vocab['[unused1]']
        self.end_token = self.vocab['[unused2]']
        self.seg_token = self.vocab['[unused3]']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(
            tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch_dev(self, doc_batch, tgt_data):

        translations = []

        # batch_size = len(doc_batch)
        batch_size = min(len(doc_batch), len(tgt_data)) # 0724修改，是否合理？

        for b in range(batch_size):

            # generated text
            pred_summ = self.tokenizer.convert_ids_to_tokens(
                [int(n) for n in doc_batch[b]])
            pred_summ = ' '.join(pred_summ)
            pred_summ = pred_summ.replace('[unused0]', '').replace('[unused1]', '').\
                replace('[unused2]', '').replace('[unused5]', '#').replace('[UNK]', '').strip()
            pred_summ = ' '.join(pred_summ.split())
            pred_summ = pred_summ.replace(" ##", "")

            gold_data = ' '.join(tgt_data[b])
            gold_data = gold_data.replace('[PAD]', '').replace('[unused1]', '').\
                replace('[unused2]', '').replace('[unused5]', '#').replace("[UNK]", '').strip()
            gold_data = ' '.join(gold_data.split())
            gold_data = gold_data.replace(" ##", "")

            translations.append((pred_summ, gold_data))

        return translations

    def from_batch_test(self, batch, output_batch, tgt_data):

        translations = []

        batch_size = len(batch)

        origin_txt, ex_segs = batch.original_str, batch.ex_segs

        ex_segs = [sum(ex_segs[:i]) for i in range(len(ex_segs)+1)]

        for b in range(batch_size):
            # original text
            original_sent = ' <S> '.join(origin_txt[ex_segs[b]:ex_segs[b+1]])

            # long doc context text
            pred_summ = self.tokenizer.convert_ids_to_tokens(
                [int(n) for n in output_batch[b]])
            pred_summ = ' '.join(pred_summ)
            pred_summ = pred_summ.replace(" ##", "")

            pred_summ = pred_summ.replace('[unused0]', '').replace('[unused1]', '').\
                replace('[unused2]', '').replace('[unused5]', '#').replace('[UNK]', '').strip()
            pred_summ = ' '.join(pred_summ.split())

            gold_data = ' '.join(tgt_data[b])
            gold_data = gold_data.replace('[PAD]', '').replace('[unused1]', '').replace('[unused2]', '').\
                replace('[unused5]', '#').replace('[UNK]', '').strip()
            gold_data = ' '.join(gold_data.split())
            gold_data = gold_data.replace(" ##", "")

            translation = (original_sent, pred_summ, gold_data)
            translations.append(translation)

        return translations

    def validate(self, data_iter, step, attn_debug=False):

        self.model.eval()
        # 三部分
        gold_path = self.args.result_path + 'step.%d.gold_temp' % step
        pred_path = self.args.result_path + 'step.%d.pred_temp' % step
        # 整体
        v_gold_path = self.args.result_path + 'step.%d.verdict_gold_temp' % step
        v_pred_path = self.args.result_path + 'step.%d.verdict_pred_temp' % step
        # 优点
        p_gold_path = self.args.result_path + 'step.%d.pros_gold_temp' % step
        p_pred_path = self.args.result_path + 'step.%d.pros_pred_temp' % step
        # 缺点
        c_gold_path = self.args.result_path + 'step.%d.cons_gold_temp' % step
        c_pred_path = self.args.result_path + 'step.%d.cons_pred_temp' % step
        
        gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        pred_out_file = codecs.open(pred_path, 'w', 'utf-8')
        v_gold_out_file = codecs.open(v_gold_path, 'w', 'utf-8')
        v_pred_out_file = codecs.open(v_pred_path, 'w', 'utf-8')
        p_gold_out_file = codecs.open(p_gold_path, 'w', 'utf-8')
        p_pred_out_file = codecs.open(p_pred_path, 'w', 'utf-8')
        c_gold_out_file = codecs.open(c_gold_path, 'w', 'utf-8')
        c_pred_out_file = codecs.open(c_pred_path, 'w', 'utf-8')

        ct = 0
        ext_acc_num = 0
        ext_pred_num = 0
        ext_gold_num = 0

        with torch.no_grad():
            for batch in data_iter:
                output_data, tgt_data, ext_pred, ext_gold = self.translate_batch(batch)
                # print("**********start*************")
                # print(output_data) # [tensor([27741, 27741, 27741, 27741,     2], device='cuda:0'), tensor([26265, 26265, 26265, 26265,     2], device='cuda:0')]
                # print(tgt_data) # ['[unused1]', 'A', 'step', 'by', 'step', 'guide', 'to', 'real', '-', 'estate', 'investment', 'based', 'on', 'current', 'market', 'trends', '.', '[unused13]', 'Reader', '##s', 'appreciate', 'the', 'modern', 'day', 'advice', 'this', 'book', 'offers', 'when', 'it', 'comes', 'to', 'real', 'estate', 'investing', 'Gen', '##uin', '##e', 'and', 'written', 'in', 'a', 'manner', 'that', 'is', 'not', 'overly', 'sales', 'oriented', 'A', 'must', '-', 'read', 'for', 'landlord', '##s', 'or', 'those', 'thinking', 'of', 'managing', 'tenants', '[unused13]', 'We', 'would', 'love', 'it', 'if', 'the', 'author', 'didn', "'", 't', 'mention', 'his', 'website', 'so', 'much', 'as', 'it', 'de', '##tracts', 'from', 'the', 'overall', 'material', '[unused2]'], ['[unused1]', 'The', 'attractive', '##ly', 'designed', 'Barnes', '&', 'Noble', "'", 's', 'No', '##ok', 'G', '##low', '##L', '##ight', 'should', 'please', 'existing', 'No', '##ok', 'and', 'B', '&', 'N', 'store', 'fans', 'alike', ',', 'although', 'some', 'may', 'not', 'like', 'its', 'reduced', 'feature', 'set', '.', '[unused13]', 'Beautiful', 'edge', '-', 'lighting', 'Class', '##y', 'design', 'Light', '##er', 'than', 'the', 'Kind', '##le', 'Paper', '##w', '##hit', '##e', 'S', '##mo', '##oth', 'reading', 'experience', 'No', 'ads', 'E', '##pu', '##b', 'support', 'In', '-', 'person', 'support', 'at', 'all', 'B', '&', 'N', 'store', 'locations', '[unused13]', 'Not', 'quite', 'as', 's', '##vel', '##te', 'or', 're', '##sp', '##ons', '##ive', 'as', 'the', 'Kind', '##le', 'Paper', '##w', '##hit', '##e', 'No', 'more', 'memory', 'card', 'slot', 'or', 'page', 'turn', 'buttons', 'Some', 'minor', 'issues', 'with', 'speed', 'and', 'font', 'rendering', '[unused2]']]
                # print(ext_pred) # [[95, 61], [74, 38]]
                # print(ext_gold) # [[35, 99, 2, 86, 94, 9], [8, 33, 57, 62, 54]]
                # print("***********end**************")
                # assert 0
                translations = self.from_batch_dev(output_data, tgt_data) # 删除起始终止符，未知符，以及原有的角色标记符
                # print("**********start*************")
                # print(translations) # [('##rrosion ##rrosion ##rrosion ##rrosion', "A step by step guide to real - estate investment based on current market trends . [unused13] Reader ##s appreciate the modern day advice this book offers when it comes to real estate investing Gen ##uin ##e and written in a manner that is not overly sales oriented A must - read for landlord ##s or those thinking of managing tenants [unused13] We would love it if the author didn ' t mention his website so much as it de ##tracts from the overall material"), ('##pticism ##pticism ##pticism ##pticism', "The attractive ##ly designed Barnes & Noble ' s No ##ok G ##low ##L ##ight should please existing No ##ok and B & N store fans alike , although some may not like its reduced feature set . [unused13] Beautiful edge - lighting Class ##y design Light ##er than the Kind ##le Paper ##w ##hit ##e S ##mo ##oth reading experience No ads E ##pu ##b support In - person support at all B & N store locations [unused13] Not quite as s ##vel ##te or re ##sp ##ons ##ive as the Kind ##le Paper ##w ##hit ##e No more memory card slot or page turn buttons Some minor issues with speed and font rendering")]
                # # translations中包含生成的摘要和黄金摘要
                # print("***********end**************")
                # print(range(len(translations))) # 2
                # assert 0


                for idx in range(len(translations)):
                    if ct % 100 == 0:
                        print("Processing %d" % ct)
                    pred_summ, gold_data = translations[idx]
                    # print("**********start*************")
                    # print(pred_summ)
                    # print(gold_data)
                    if "[unused13]" in pred_summ:
                        split_pred = pred_summ.split("[unused13]")
                        verdict_pred = split_pred[0].strip()
                        if len(split_pred) == 2:
                            pros_pred = split_pred[1].strip()
                            cons_pred = " "
                        else:
                            pros_pred = split_pred[1].strip()
                            cons_pred = split_pred[2].strip() + " "
                    else:
                        verdict_pred = pred_summ.strip()
                        pros_pred = " "
                        cons_pred = " "

                    if "[unused13]" in gold_data:
                        split_gold = gold_data.split("[unused13]")
                        verdict_gold = split_gold[0].strip()
                        if len(split_gold) == 2:
                            pros_gold = split_gold[1].strip()
                            cons_gold = " "
                        else:
                            pros_gold = split_gold[1].strip()
                            cons_gold = split_gold[2].strip() + " "
                    else:
                        verdict_gold = gold_data.strip()
                        pros_gold = " "
                        cons_gold = " "

                    # print("Verdict_pred: ", verdict_pred)
                    # print("Pros_pred: ", pros_pred)
                    # print("Cons_pred: ", cons_pred)
                    # print("Verdict_gold: ", verdict_gold)
                    # print("Pros_gold: ", pros_gold)
                    # print("Cons_gold: ", cons_gold)
                    # print("***********end**************")
                    # assert 0

                    # ext f1 calculate
                    acc_num = len(ext_pred[idx] + ext_gold[idx]) - len(set(ext_pred[idx] + ext_gold[idx]))
                    pred_num = len(ext_pred[idx])
                    gold_num = len(ext_gold[idx])
                    ext_acc_num += acc_num
                    ext_pred_num += pred_num
                    ext_gold_num += gold_num
                    pred_out_file.write(pred_summ + '\n')
                    gold_out_file.write(gold_data + '\n')
                    v_pred_out_file.write(verdict_pred + '\n')
                    v_gold_out_file.write(verdict_gold + '\n')
                    p_pred_out_file.write(pros_pred + '\n')
                    p_gold_out_file.write(pros_gold + '\n')
                    c_pred_out_file.write(cons_pred + '\n')
                    c_gold_out_file.write(cons_gold + '\n')
                    ct += 1
                pred_out_file.flush()
                gold_out_file.flush()
                v_pred_out_file.flush()
                v_gold_out_file.flush()
                p_pred_out_file.flush()
                p_gold_out_file.flush()
                c_pred_out_file.flush()
                c_gold_out_file.flush()

        pred_out_file.close()
        gold_out_file.close()
        v_pred_out_file.close()
        v_gold_out_file.close()
        p_pred_out_file.close()
        p_gold_out_file.close()
        c_pred_out_file.close()
        c_gold_out_file.close()

        if (step != -1):
            pred_bleu = test_bleu(pred_path, gold_path)
            file_rouge = FilesRouge(hyp_path=pred_path, ref_path=gold_path)
            pred_rouges = file_rouge.get_scores(avg=True)
            v_file_rouge = FilesRouge(hyp_path=v_pred_path, ref_path=v_gold_path)
            v_pred_rouges = v_file_rouge.get_scores(avg=True)
            p_file_rouge = FilesRouge(hyp_path=p_pred_path, ref_path=p_gold_path)
            p_pred_rouges = p_file_rouge.get_scores(avg=True)
            c_file_rouge = FilesRouge(hyp_path=c_pred_path, ref_path=c_gold_path)
            c_pred_rouges = c_file_rouge.get_scores(avg=True)

            f1, p, r = test_f1(ext_acc_num, ext_pred_num, ext_gold_num)
            self.logger.info('Ext Sent Score at step %d: \n>> P/R/F1: %.2f/%.2f/%.2f' %
                             (step, p*100, r*100, f1*100))
            self.logger.info('Gold Length at step %d: %.2f' %
                             (step, test_length(gold_path, gold_path, ratio=False)))
            self.logger.info('Prediction Length ratio at step %d: %.2f' %
                             (step, test_length(pred_path, gold_path)))
            self.logger.info('Prediction Bleu at step %d: %.2f' %
                             (step, pred_bleu*100))
            self.logger.info('Prediction Rouges at step %d: \n%s\nverdict:\n%s\npros:\n%s\ncons:\n%s\n' %
                             (step, rouge_results_to_str(pred_rouges), rouge_results_to_str(v_pred_rouges), rouge_results_to_str(p_pred_rouges), rouge_results_to_str(c_pred_rouges)))
            rouge_results = (pred_rouges["rouge-1"]['f'],
                             pred_rouges["rouge-2"]['f'],
                             pred_rouges["rouge-l"]['f'])
            v_rouge_results = (v_pred_rouges["rouge-1"]['f'],
                               v_pred_rouges["rouge-2"]['f'],
                               v_pred_rouges["rouge-l"]['f'])
            p_rouge_results = (p_pred_rouges["rouge-1"]['f'],
                               p_pred_rouges["rouge-2"]['f'],
                               p_pred_rouges["rouge-l"]['f'])
            c_rouge_results = (c_pred_rouges["rouge-1"]['f'],
                               c_pred_rouges["rouge-2"]['f'],
                               c_pred_rouges["rouge-l"]['f'])
        self.model.train() # 训练过程中评估，模式需要修改回去，否则loss报错
        return rouge_results, v_rouge_results, p_rouge_results, c_rouge_results

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        output_path = self.args.result_path + '.%d.output' % step
        output_file = codecs.open(output_path, 'w', 'utf-8')
        
        gold_path = self.args.result_path + '.%d.gold_test' % step
        pred_path = self.args.result_path + '.%d.pred_test' % step
        gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        pred_out_file = codecs.open(pred_path, 'w', 'utf-8')

        v_gold_path = self.args.result_path + '.%d.verdict_gold_test' % step
        v_pred_path = self.args.result_path + '.%d.verdict_pred_test' % step
        v_gold_out_file = codecs.open(v_gold_path, 'w', 'utf-8')
        v_pred_out_file = codecs.open(v_pred_path, 'w', 'utf-8')

        p_gold_path = self.args.result_path + '.%d.pros_gold_test' % step
        p_pred_path = self.args.result_path + '.%d.pros_pred_test' % step
        p_gold_out_file = codecs.open(p_gold_path, 'w', 'utf-8')
        p_pred_out_file = codecs.open(p_pred_path, 'w', 'utf-8')

        c_gold_path = self.args.result_path + '.%d.cons_gold_test' % step
        c_pred_path = self.args.result_path + '.%d.cons_pred_test' % step
        c_gold_out_file = codecs.open(c_gold_path, 'w', 'utf-8')
        c_pred_out_file = codecs.open(c_pred_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []

        ct = 0
        ext_acc_num = 0
        ext_pred_num = 0
        ext_gold_num = 0

        with torch.no_grad():
            rouge = Rouge()
            for batch in data_iter:
                output_data, tgt_data, ext_pred, ext_gold = self.translate_batch(batch)
                translations = self.from_batch_test(batch, output_data, tgt_data)

                for idx in range(len(translations)):
                    origin_sent, pred_summ, gold_data = translations[idx]
                    # 将生成摘要和黄金摘要都分为三个部分
                    if "[unused13]" in pred_summ:
                        split_pred = pred_summ.split("[unused13]")
                        verdict_pred = split_pred[0].strip()
                        if len(split_pred) == 2:
                            pros_pred = split_pred[1].strip()
                            cons_pred = " "
                        else:
                            pros_pred = split_pred[1].strip()
                            cons_pred = split_pred[2].strip() + " "
                    else:
                        verdict_pred = pred_summ.strip()
                        pros_pred = " "
                        cons_pred = " "

                    if "[unused13]" in gold_data:
                        split_gold = gold_data.split("[unused13]")
                        verdict_gold = split_gold[0].strip()
                        if len(split_gold) == 2:
                            pros_gold = split_gold[1].strip()
                            cons_gold = " "
                        else:
                            pros_gold = split_gold[1].strip()
                            cons_gold = split_gold[2].strip() + " "
                    else:
                        verdict_gold = gold_data.strip()
                        pros_gold = " "
                        cons_gold = " "

                    if ct % 100 == 0:
                        print("Processing %d" % ct)
                    output_file.write("ID      : %d\n" % ct)
                    output_file.write("ORIGIN  : \n    " + origin_sent.replace('<S>', '\n    ') + "\n")
                    output_file.write("GOLD    : " + gold_data.strip() + "\n")
                    output_file.write("DOC_GEN : " + pred_summ.strip() + "\n")
                    rouge_score = rouge.get_scores(pred_summ, gold_data)
                    # v_rouge_score = rouge.get_scores(verdict_pred, verdict_gold)
                    # p_rouge_score = rouge.get_scores(pros_pred, pros_gold)
                    # c_rouge_score = rouge.get_scores(cons_pred, cons_gold)
                    bleu_score = sentence_bleu([gold_data.split()], pred_summ.split(),
                                               smoothing_function=SmoothingFunction().method1)
                    output_file.write("DOC_GEN  bleu & rouge-f 1/2/l:    %.4f & %.4f/%.4f/%.4f\n" %
                                      (bleu_score, rouge_score[0]["rouge-1"]["f"],
                                       rouge_score[0]["rouge-2"]["f"], rouge_score[0]["rouge-l"]["f"]))
                    # output_file.write("verdict:rouge-f 1/2/l:    %.4f/%.4f/%.4f\n" %
                    #                   (v_rouge_score[0]["rouge-1"]["f"], v_rouge_score[0]["rouge-2"]["f"], v_rouge_score[0]["rouge-l"]["f"]))
                    # output_file.write("pros:rouge-f 1/2/l:    %.4f/%.4f/%.4f\n" %
                    #                   (p_rouge_score[0]["rouge-1"]["f"], p_rouge_score[0]["rouge-2"]["f"], p_rouge_score[0]["rouge-l"]["f"]))
                    # output_file.write("cons:rouge-f 1/2/l:    %.4f/%.4f/%.4f\n" %
                    #                   (c_rouge_score[0]["rouge-1"]["f"], c_rouge_score[0]["rouge-2"]["f"], c_rouge_score[0]["rouge-l"]["f"]))
                    
                    # ext f1 calculate
                    acc_num = len(ext_pred[idx] + ext_gold[idx]) - len(set(ext_pred[idx] + ext_gold[idx]))
                    pred_num = len(ext_pred[idx])
                    gold_num = len(ext_gold[idx])
                    ext_acc_num += acc_num
                    ext_pred_num += pred_num
                    ext_gold_num += gold_num
                    f1, p, r = test_f1(acc_num, pred_num, gold_num)
                    output_file.write("EXT_GOLD: [" + ','.join([str(i) for i in sorted(ext_gold[idx])]) + "]\n")
                    output_file.write("EXT_PRED: [" + ','.join([str(i) for i in sorted(ext_pred[idx])]) + "]\n")
                    output_file.write("EXT_SCORE  P/R/F1:    %.4f/%.4f/%.4f\n\n" % (p, r, f1))
                    pred_out_file.write(pred_summ.strip() + '\n')
                    gold_out_file.write(gold_data.strip() + '\n')
                    v_pred_out_file.write(verdict_pred + '\n')
                    v_gold_out_file.write(verdict_gold + '\n')
                    p_pred_out_file.write(pros_pred + '\n')
                    p_gold_out_file.write(pros_gold + '\n')
                    c_pred_out_file.write(cons_pred + '\n')
                    c_gold_out_file.write(cons_gold + '\n')
                    ct += 1
                pred_out_file.flush()
                gold_out_file.flush()
                output_file.flush()
                v_pred_out_file.flush()
                v_gold_out_file.flush()
                p_pred_out_file.flush()
                p_gold_out_file.flush()
                c_pred_out_file.flush()
                c_gold_out_file.flush()

        pred_out_file.close()
        gold_out_file.close()
        output_file.close()
        v_pred_out_file.close()
        v_gold_out_file.close()
        p_pred_out_file.close()
        p_gold_out_file.close()
        c_pred_out_file.close()
        c_gold_out_file.close()

        if (step != -1):
            pred_bleu = test_bleu(pred_path, gold_path)
            file_rouge = FilesRouge(hyp_path=pred_path, ref_path=gold_path)
            pred_rouges = file_rouge.get_scores(avg=True)
            v_file_rouge = FilesRouge(hyp_path=v_pred_path, ref_path=v_gold_path)
            v_pred_rouges = v_file_rouge.get_scores(avg=True)
            p_file_rouge = FilesRouge(hyp_path=p_pred_path, ref_path=p_gold_path)
            p_pred_rouges = p_file_rouge.get_scores(avg=True)
            c_file_rouge = FilesRouge(hyp_path=c_pred_path, ref_path=c_gold_path)
            c_pred_rouges = c_file_rouge.get_scores(avg=True)

            f1, p, r = test_f1(ext_acc_num, ext_pred_num, ext_gold_num)
            self.logger.info('Ext Sent Score at step %d: \n>> P/R/F1: %.2f/%.2f/%.2f' %
                             (step, p*100, r*100, f1*100))
            self.logger.info('Gold Length at step %d: %.2f' %
                             (step, test_length(gold_path, gold_path, ratio=False)))
            self.logger.info('Prediction Length ratio at step %d: %.2f' %
                             (step, test_length(pred_path, gold_path)))
            self.logger.info('Prediction Bleu at step %d: %.2f' %
                             (step, pred_bleu*100))
            self.logger.info('Prediction Rouges at step %d: \n%s\nverdict:\n%s\npros:\n%s\ncons:\n%s\n' %
                             (step, rouge_results_to_str(pred_rouges), rouge_results_to_str(v_pred_rouges), rouge_results_to_str(p_pred_rouges), rouge_results_to_str(c_pred_rouges)))

    def translate_batch(self, batch):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        if self.args.pretrain:
            pn_result, _, _, output_data = self.model.pretrain(batch)
        else:
            _, _, _, output_data, pn_result = self.model(batch)
        tgt_txt = batch.tgt_txt
        gold_ext = batch.tgt_labels
        return output_data, tgt_txt, pn_result, gold_ext


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
