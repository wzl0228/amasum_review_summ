# amasum_review_summ
1. Convert dataset format

    ```
    python dataset_format_convert.py -data_path amasum_raw_dataset/test -mode test
    
    python dataset_format_convert.py -data_path amasum_raw_dataset/train -mode train
    
    python dataset_format_convert.py -data_path amasum_raw_dataset/valid -mode valid
    ```

2. Pre-train word2vec embeddings

    ```
    python train_emb.py
    ```

3. Data Processing

	```
	python preprocess.py -tokenize -truncated -add_ex_label
 
    注意：不要重复运行，会使bert_data/idf_info.pt文件清空出问题。如果需要重新得到bert_data下的pt文件，先删除bert_data下的所有内容，再运行此步骤
	```

4. Pre-train the pipeline model (Ext + Abs)

	```
	python train.py -data_path bert_data/ali -log_file logs/pipeline.topic.train.log -sep_optim -topic_model -split_noise -pretrain -model_path checkpoints/pipeline_topic

    python train.py -data_path bert_data/ali -log_file logs/01pipeline.topic.train.log -sep_optim -split_noise -pretrain -model_path checkpoints/01pipeline_topic -result_path results/01 -topic_model True -agent True -cust True
    
	```

5. Train the whole model with RL

    ```
    python train.py -data_path bert_data/ali -log_file logs/rl.topic.train.log -model_path checkpoints/rl_topic -topic_model -split_noise -train_from checkpoints/pipeline_topic/model_step_48000.pt -train_from_ignore_optim -lr 0.00001 -save_checkpoint_steps 500 -train_steps 30000 -result_path results/trainrl
    ```

6. Validate

	```
	python train.py -mode validate -data_path bert_data/ali -log_file logs/rl.topic.val.log -alpha 0.95 -model_path checkpoints/rl_topic -topic_model -split_noise -result_path results/val
	```

7. Test

	```
	python train.py -mode test -data_path bert_data/ali -test_from checkpoints/rl_topic/02model_step_500.pt -log_file logs/rl.topic.test.log -alpha 0.95 -topic_model -split_noise -result_path results/test
	```