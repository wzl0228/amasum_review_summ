# amasum_review_summ
代码部分参考：[https://github.com/RowitZou/topic-dialog-summ](https://github.com/RowitZou/topic-dialog-summ)

[我的实验记录](https://github.com/wzl0228/note/blob/main/%E4%B8%A4%E9%98%B6%E6%AE%B5%E6%84%8F%E8%A7%81%E6%91%98%E8%A6%81%E6%A8%A1%E5%9E%8B%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95.md)

## 1. 数据集格式转换

```
python dataset_format_convert.py -data_path amasum_raw_dataset/test -mode test
python dataset_format_convert.py -data_path amasum_raw_dataset/train -mode train
python dataset_format_convert.py -data_path amasum_raw_dataset/valid -mode valid
```

## 2. 预训练word2vec

```
python train_emb.py
```

## 3. 数据预处理

```
python preprocess.py -tokenize -truncated -add_ex_label
```
    
**注意：不要重复运行，会使bert_data/idf_info.pt文件清空出问题。如果需要重新得到bert_data下的pt文件，先删除bert_data下的所有内容，再运行此步骤**

## 4. 训练模型pipeline (Ext + Abs)

```
python train.py -data_path bert_data/ali -idf_info_path bert_data/idf_info.pt -log_file logs/01pipeline.topic.train.log -sep_optim -split_noise -pretrain -model_path checkpoints/01pipeline_topic -result_path results/01 -topic_model True -agent True -cust True
```
    
可以同时提交多次训练，修改序号01/02/03...，提高效率。
数据预处理不同时，可以修改路径：bert_data_2/ali -idf_info_path bert_data_2/idf_info.pt
    
```
python train.py -data_path bert_data_2/ali -idf_info_path bert_data_2/idf_info.pt -log_file logs/02pipeline.topic.train.log -sep_optim -split_noise -pretrain -model_path checkpoints/02pipeline_topic -result_path results/02 -topic_model True -agent True -cust True
```

## 5. 强化学习模型

```
python train.py -data_path bert_data/ali -idf_info_path bert_data/idf_info.pt -log_file logs/01rl.topic.train.log -model_path checkpoints/01rl_topic -topic_model True -agent True -cust True -split_noise -train_from checkpoints/01pipeline_topic/model_step_XXXX.pt -train_from_ignore_optim -lr 0.00001 -save_checkpoint_steps 500 -train_steps 30000 -result_path results/01trainrl
```
```
python train.py -data_path bert_data_2/ali -idf_info_path bert_data_2/idf_info.pt -log_file logs/02rl.topic.train.log -model_path checkpoints/02rl_topic -topic_model True -agent True -cust True -split_noise -train_from checkpoints/02pipeline_topic/model_step_XXXX.pt -train_from_ignore_optim -lr 0.00001 -save_checkpoint_steps 500 -train_steps 30000 -result_path results/02trainrl
```

## 6. 测试
模型在训练过程中会进行验证（在验证集上计算指标）
等待模型训练结束，进行模型测试（在测试集上计算指标）
```
python train.py -mode test -data_path bert_data/ali -idf_info_path bert_data/idf_info.pt -test_from checkpoints/01rl_topic/model_step_XXXX.pt -log_file logs/01rl.topic.test.log -alpha 0.95 -topic_model -split_noise -result_path results/01test
```
```
python train.py -mode test -data_path bert_data_2/ali -idf_info_path bert_data_2/idf_info.pt -test_from checkpoints/02rl_topic/model_step_XXXX.pt -log_file logs/02rl.topic.test.log -alpha 0.95 -topic_model -split_noise -result_path results/02test
```

数据：
[amasum数据集](https://github.com/abrazinskas/SelSum/tree/master/data)