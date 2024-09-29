import torch
import numpy as np
from transformers import AutoTokenizer,AutoModel
import torch.nn as nn
# from torch.utils.data import DataLoader,Dataset
# import csv
import os
import json
import glob
import config_fea1
import numpy
import config
# from sklearn import metrics
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def judge_type(instr_token):
    if instr_token in config_fea1.DataTransferInstr:
        token_type = 1
    elif instr_token in config_fea1.ArithmeticInstr:
        token_type = 2
    elif instr_token in config_fea1.LogicalInstr:
        token_type = 3
    elif instr_token in config_fea1.ControlFlowInstr:
        token_type = 4
    elif instr_token in config_fea1.BitInstr:
        token_type = 5
    elif instr_token in config_fea1.ConditionalTransferInstr:
        token_type = 6
    elif instr_token in config_fea1.ConditionalSettingInstr:
        token_type = 7
    elif instr_token in config_fea1.StackInstr:
        token_type = 8
    elif instr_token in config_fea1.DataConversionInstr:
        token_type = 9
    elif instr_token in config_fea1.CompareInstr:
        token_type = 10
    elif instr_token in config_fea1.REG_list:
        token_type = 11
    elif instr_token in config_fea1.IMM_list:
        token_type = 12
    elif instr_token in config_fea1.MEM_list:
        token_type = 13
    # elif instr_token in config_fea1.NEAR_list:
    #     token_type = 14
    else:
        token_type = 0
        # if instr_token == "[CLS]" or instr_token == "[SEP]" or instr_token == "[PAD]" or instr_token == "\n":
        #     token_type = 0
        # else:
        #     print(instr_token)
        #     print("token miss!!!!")
    return token_type


def gener_instr_type(input_instrs):

    instr_type_lists = list()
    input_instr_list = input_instrs.split(' ')
    instr_type_list = numpy.zeros(512, dtype=np.int32)

    flag = 1
    for instr_token in input_instr_list:
        if flag == 512:
            break
        ce = judge_type(instr_token)
        instr_type_list[flag] = ce
        flag += 1

    instr_type_lists.append(instr_type_list)
    return torch.from_numpy(np.array(instr_type_lists))



def get_sentence_wec(tokenizer, func_sentences, model, sentences_num):
    func_vec = []
    sentence_num = 64    #process 64 sentences for the largest against 'cuda out of memory'
    # if len(func_sentences)>sentence_num:
    #     print('==function partitioned==')


    for i in range(0,len(func_sentences),sentence_num):
        sents = func_sentences[i:i+sentence_num]
        input_ids = []
        attention_mask = []
        type_ids = []

        for sent in sents:
            sent = sent.strip()
            sent_type = gener_instr_type(sent)
            # print(sent)
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                # padding=True,
                # pad_to_max_length = True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation = True
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_mask.append(encoded_dict['attention_mask'])
            type_ids.append(sent_type)

        input_ids = torch.cat(input_ids,dim=0).to(device)
        attention_mask = torch.cat(attention_mask,dim=0).to(device)
        type_idx = torch.cat(type_ids, dim=0).to(device)

        new_input_ids = [input_ids, type_idx]
        with torch.no_grad():

            # outputs = model(input_ids,attention_mask)
            outputs = model(new_input_ids, attention_mask)

            # # pooler-output as vec:
            # pooler_output = outputs[1].cpu().numpy()
            # func_vec.extend(pooler_output)

            # # last layer as vec
            # last_layer = torch.mean(outputs[0],1).cpu().numpy()
            # func_vec.extend(last_layer)

            #last 4 layer as vec
            hidden_states = outputs.hidden_states
            # sentence_embeddings = torch.mean(torch.stack(hidden_states[-4:]),0)
            #
            # for i in len(type_ids):
            #     sentence_embedding = sentence_embeddings[i]
            #
            #     for j in type_ids[i][0]:
            #         if j == 0:
            #             continue
            #
            # for sentence_embedding in sentence_embeddings:

            layer_mean = torch.mean(torch.mean(torch.stack(hidden_states[-4:]),0),1).cpu().numpy()
            func_vec.extend(layer_mean)

    return func_vec


def instr_normalization(func_sentences):

    norma_sentences = list()
    type_sentences = list()

    for sentences in func_sentences:
        sentence_list = sentences.split(' ')

        res = ''
        norma_sentence_list = list()
        type_sentence_list = list()
        type_sentence_list.append(0)

        for sentence in sentence_list:
            sentence1 = sentence.split('~')
            norma_sentence_list.append(sentence1[0])
            res += sentence1[0] + ' '
            type_sentence_list.append(1)

            if len(sentence1) > 1:
                operator_list = sentence1[1].split('>')
                for operator in operator_list:
                    norma_sentence_list.append(operator.split('=')[1])
                    res += operator.split('=')[1] + ' '
                    type_sentence_list.append(2)
        # norma_sentences.append(norma_sentence_list)
        type_sentence_list.append(0)
        type_sentences.append(type_sentence_list)
        norma_sentences.append(res)

    return norma_sentences,type_sentences

def instr_normalization_old(func_sentences):

    norma_sentences = list()
    type_sentences = list()

    for sentences in func_sentences:
        sentence_list = sentences.split(' ')

        res = ''
        norma_sentence_list = list()
        type_sentence_list = list()
        type_sentence_list.append(0)

        for sentence in sentence_list:
            sentence1 = sentence.split('~')
            norma_sentence_list.append(sentence1[0])
            res += sentence1[0] + ' '
            type_sentence_list.append(1)

            if len(sentence1) > 1:
                operator_list = sentence1[1].split('>')
                for operator in operator_list:
                    norma_sentence_list.append(operator.split('-')[2])
                    res += operator.split('-')[2] + ' '
                    type_sentence_list.append(2)
        # norma_sentences.append(norma_sentence_list)
        type_sentence_list.append(0)
        type_sentences.append(type_sentence_list)
        norma_sentences.append(res)

    return norma_sentences,type_sentences


model_type = config.STEP2_TOKENIER_DIR

software_name = config.STEP2_SOFTWARE_NAME

read_inst_dir = config.STEP2_INSTRUCTIONS_DIR + os.sep + software_name

save_dir = config.STEP2_SAVE_DIR + os.sep + software_name

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
tokenizer = AutoTokenizer.from_pretrained(model_type)

arch_paths = glob.glob(read_inst_dir+ os.sep + '*')
for arch_path in arch_paths:
    cur_file = '_'.join(arch_path.split(os.sep)[-1].split('_')[:4])
    print('\ncur_file:', cur_file)
    # cur_model_path = ''
    # model_paths = glob.glob('../finetune_bert/data/model_save' + os.sep + software_name + os.sep + cur_file[:-5] +'*')
    # model_paths = glob.glob('../finetune_bert/data/model_save' + os.sep + 'coreutils' + os.sep + cur_file + '*')
    # print('using model path:',model_paths[0])
    bert_model = AutoModel.from_pretrained(config.STEP2_MODEL_DIR, output_hidden_states = True)
    # bert_model = AutoModel.from_pretrained(model_type,output_hidden_states = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    if torch.cuda.device_count() > 1:
        bert_model = torch.nn.DataParallel(bert_model)

    bert_model.to(device)
    bert_model.eval()

    json_name = arch_path.split(os.sep)[-1][:-5]
    save_path = save_dir + os.sep + json_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # cnt = 1
    with open(arch_path,'r') as fp:
        for line in fp.readlines():     #line:func information
            infos = json.loads(line)    #turn to the type of dictionary
            func_sentences = []

            # if infos['n_num']<5:
            #     continue
            #
            # flag = 0
            # for b_info in infos['features']:
            #     if len(b_info[12]) == 0:
            #         flag = 1
            #         break
            # if flag:
            #     continue

            for info in infos['features']:  #info : feature of a block
                sentence = ' '.join(info[11])   #get standardized instruction of a block
                func_sentences.append(sentence)

            func_sentences, type_sentences = instr_normalization(func_sentences)

            func_vec = get_sentence_wec(tokenizer,func_sentences,bert_model, type_sentences)
            fname = infos['fname']
            save_file_path = save_path + os.sep + fname
            np.save(save_file_path,func_vec)
            # print('function{}:'.format(cnt),fname)
            # cnt = cnt+1



