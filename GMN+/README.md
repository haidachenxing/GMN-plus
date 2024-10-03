# GMN+: A Binary Homologous Vulnerability Detection Method Based on Graph Matching Neural Network with Enhanced Attention

## Prerequiste

Make sure you have installed all of following packages or libraries (including dependencies if necessary) in your workspace:

1. Python3
2. Pytorch
3. scikit-learn
4. Numpy
5. IDA pro

## Dataset

- Glpk (v4.65)
- Xml (v2.9.4)
- Sqlite3 (v0.8.6)

# Instruction Preprocessor and Semantic Learner

1. Run token/gen_fit_all_inst.py, extract instructions from json file.
2. Run token/extract_token.py, extract tokens from instrcutions. Tokens are divided into operators and operadns.
3. Run token/build_mytokenizer.py, build new tokenzier.Adding new tokens into tokenizer.
4. Run token/gen_label_instr.py, generate training data.
5. Run mlm_finetune.py, train your model.
6. Run gen_FIT_sent_bert_vec_means.py, inference model generates semantic embeddings.

Notice: exchange modeling_bert in origin Transformer.

# Graph Learner

1. In configure.py, config features_train_path which is your dataset path，and other parameters
2. Run train.py, train your model.