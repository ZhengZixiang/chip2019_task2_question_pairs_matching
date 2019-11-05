# chip2019_task2_question_pairs_matching
[CHIP 2019 平安医疗科技疾病问答迁移学习比赛](https://www.biendata.com/competition/chip2019/)，本质上就是一个类似于Quora Question Pairs的问句匹配问题。基于[huggingface/pytorch-transformers](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py)实现的BERT baseline，代码比较冗余，中文预训练模型采用[ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)。因为条件有限（没有GPU。。。），所以只跑了几个baseline（提交次数惨淡），没有trick、模型融合以及超参数选择，只做10折交叉验证，A榜应该就能达到0.878+，B榜0.864+，A榜rank 9，B榜rank11，剔除了小号和未报名队伍后rank7，作为baseline效果还是可以的。

项目文件目录结构及文件说明如下：
```
|-- README.md
|-- data_augmentation.py 利用问句相似性传递进行数据增强和生成10折交叉验证数据文件
|-- data_utils.py 读取数据文件，转化为模型输入等操作
|-- extract_bert_char_embedding.py 抽取出BERT的字向量，然后用在例如ESIM之类的常规模型上，效果不好
|-- feature_engineering.py 对每一折数据抽取出数据字词级别的tfidf等特征
|-- model_qpm.py BERT常规句对分类
|-- model_multitask.py 在qpm的基础上，增加一个子任务，将句对句子合在一起，判断所属类目
|-- model_feature.py 在qpm的基础上，加入手工特征的dense层
|-- model_final.py 将qpm、判断类目和手工特征三种做法融合
|-- model_final_2.py 和model_final.py基本一致，只是改变了模型保存方式
|-- post_processing.py 利用问句相似性传递进行后处理文件
|-- data 按照10折交叉验证分别存放10折数据文件
    |-- noextension 未数据增强的文件
        |-- 0
        |-- ...
    |-- THUOCL_medical.txt 清华开源的医疗词库，用于jieba加载后分词做词特征抽取
|-- tmp 按照10折交叉验证分别保存模型的目录
    |-- 0
    |-- ...
```
`model_final.py`和`model_final_2.py`只是在模型保存方式上有区别，前者占空间，后者费时间。

训练模型请使用
```
python3 model_final_2.py --model_name_or_path ./chinese_roberta_wwm_ext_pytorch/ --do_train --do_lower_case --data_dir ./data/noextension/ --max_seq_length 128 --per_gpu_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 5.0 --output_dir ./tmp/ --overwrite_output_dir --evaluate_during_training
```

预测结果请使用
```
python3 model_final_2.py --model_name_or_path ./chinese_roberta_wwm_ext_pytorch/ --do_predict --do_lower_case --data_dir ./data/noextension/ --per_gpu_test_batch_size 16 --output_dir ./tmp/
```

说说模型效果：
- 使用`RoBERTa-wwm-ext`和`RoBERTa-wwm-ext-large`能比`BERT-wwm-ext`提升0.005，而roberta base和large效果差别不大，就0.002左右。
- 利用问句相似性传递做数据增强容易过拟合，而且本训练集标注有很多问题，不对训练集做任何修改的话，用了数据增强反而下降0.005左右。同样做相似性传递后处理也会下降0.001~0.005，所以就没用数据增强和后处理。
- 加入句子分类的loss，能提升0.001。
- 加入特征工程的dense层，效果不稳定，可能是我的特征选得不好。

另外两个句子分别通过BERT得到representation后，互相做一下attention拼接到句子对的[CLS]输出后也能提升模型效果，不过后期没GPU了，没基于roberta-wwm训练新模型，所以最终提交的还是roberta_final_2.py跑出来的结果。
