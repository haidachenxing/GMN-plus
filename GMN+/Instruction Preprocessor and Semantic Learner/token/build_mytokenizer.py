from transformers import BertForMaskedLM, BertTokenizer
import os

if __name__ == '__main__':
    add_vocal = "./corpus_new/sqlite3/corpus.txt"
    read_dir = "./new_vo/sqlite3_one"
    save_dir = "./new_vo/sqlite3_one"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(add_vocal,'r') as fp:
        read_tokens = fp.readlines()
        add_tokens = []
        for i in read_tokens:
            add_tokens.append(i.strip())
        tokenizer = BertTokenizer.from_pretrained(read_dir)
        model = BertForMaskedLM.from_pretrained(read_dir)
        tokenizer.add_tokens(add_tokens)
        # blocktokens = []
        # for i in range(200):
        #     blocktokens.append('start@{}'.format(i))
        # # for i in range(200):
        # #     blocktokens.append('end@{}'.format(i))
        # for i in range(200):
        #     blocktokens.append('resultlv@{}'.format(i))
        # for i in range(200):
        #     blocktokens.append('lvname@{}'.format(i))
        # for i in range(200):
        #     blocktokens.append('lvar@{}'.format(i))

        # tokenizer.add_tokens(blocktokens)
        # tokenizer.add_special_tokens({'additional_special_tokens':['[JUMP]','[DATA]']})
        model.resize_token_embeddings(len(tokenizer))
        # tokenizer.save_pretrained('mytokenizer')
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)

