import random
import os
import time
import csv
import glob

# sample_num = 2500

# ori_insr_dir = "./data/std_inst_files/sqlite3"
ori_insr_dir = "./corpus/std_inst_files"
# save_dir = "./labled_instr/sqlite3"
save_dir = "./corpus/labled_instr"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

save_name = 'all_data.csv'

def Get_Inst_Sentences(filename):
    with open(ori_insr_dir + os.sep+filename, 'r') as fp:
        sentences = fp.readlines()
    return sentences



if __name__ == '__main__':

    sum_num = 0

    add_num = 0
    ce = 0
    nnnum = 10
    max_num = 512

    max_str = ''

    for instr_file in os.listdir(ori_insr_dir):

        ori_instr_sentences = Get_Inst_Sentences(instr_file)

        with open(ori_insr_dir + os.sep + instr_file, 'r') as fp:
            sentences = fp.readlines()

        with open(save_dir + os.sep + save_name, 'a') as sv_fp:
            # writer = csv.writer(sv_fp)

            for basic_instr in ori_instr_sentences:

                result_str = ''
                result_list = list()

                insts = basic_instr.split(' ')
                for inst in insts:
                    inst_list1 = inst.split('~')
                    operand = inst_list1[0]
                    if operand == '\n':
                        break
                    if operand == 'sync' or operand == 'UND' or operand == "NOP":
                        continue
                    result_str += operand + ' '
                    result_list.append(operand)

                    if len(inst_list1) == 1:
                        continue

                    operator_list = inst_list1[1].split('>')

                    for operatores in operator_list:
                        if operatores == '':
                            break
                        op_type = operatores.split('=')[0]
                        inst_arch = operatores.split('=')[1]
                        operator = operatores.split('=')[2]
                        result_str += operator + ' '
                        result_list.append(operator)

                if len(result_list) < nnnum:
                    add_num += 1
                    continue
                if len(result_list) >= 512:
                    ce += 1
                    continue


                result_str = result_str.replace('\n', '')

                sv_fp.write(result_str + '\n')
                sum_num += 1

            # [R5+offset]LDR
            # for i in range(sample_num):
            #     random_sent = ''
            #     while len(random_sent) < 15:
            #         random_sent = random.choice(oir_instr_sentences).strip()
            #
            #     result_str = ""
            #
            #     insts = random_sent.split(' ')
            #     for inst in insts:
            #         inst_list1 = inst.split('~')
            #         operand = inst_list1[0]
            #         result_str += operand + ' '
            #         if operand == '\n':
            #             break
            #         if len(inst_list1) == 1:
            #             continue
            #
            #         operator_list = inst_list1[1].split('_')
            #
            #         for operatores in operator_list:
            #             if operatores == '':
            #                 break
            #             op_type = operatores.split('-')[0]
            #             operator = operatores.split('-')[1]
            #             result_str += operator + ' '
            #
            #     writer.writerow([result_str])
            #
            #     result_str_num = len(result_str.split(' '))
            #     if result_str_num > nnnum:
            #         add_num += 1
            #
            #     if result_str_num > max_token_num:
            #         max_token_num = result_str_num
            #         max_str = result_str

    # print (max_token_num)
    print (sum_num)
    print (add_num)
    print (ce)
    print (float(add_num) / float(sum_num))