import os
import config_fea1

DataTransferInstr = list()
ArithmeticInstr = list()
LogicalInstr = list()
ControlFlowInstr = list()
BitInstr = list()
ConditionalTransferInstr = list()
ConditionalSettingInstr = list()
StackInstr = list()
DataConversionInstr = list()
CompareInstr = list()

REG_list = list()
IMM_list = list()
MEM_list = list()
NEAR_list = list()
other_list = list()

operator_dict = {
    1:REG_list,
    2:IMM_list,
    3:MEM_list,
    4:other_list
}

REG_operator_token = []
IMM_operator_token = ["immval"]
MEM_operator_token = []

all_token = list()
operand_token = list()


def judge_operand(operand):

    all_token.append(operand)

    if operand in config_fea1.DataTransferInstr:
        DataTransferInstr.append(operand)
    elif operand in config_fea1.ArithmeticInstr:
        ArithmeticInstr.append(operand)
    elif operand in config_fea1.LogicalInstr:
        LogicalInstr.append(operand)
    elif operand in config_fea1.ControlFlowInstr:
        ControlFlowInstr.append(operand)
    elif operand in config_fea1.BitInstr:
        BitInstr.append(operand)
    elif operand in config_fea1.ConditionalTransferInstr:
        ConditionalTransferInstr.append(operand)
    elif operand in config_fea1.ConditionalSettingInstr:
        ConditionalSettingInstr.append(operand)
    elif operand in config_fea1.StackInstr:
        StackInstr.append(operand)
    elif operand in config_fea1.DataConversionInstr:
        DataConversionInstr.append(operand)
    elif operand in config_fea1.CompareInstr:
        CompareInstr.append(operand)
    else:
        operand_token.append(operand)




if __name__ == '__main__':
    # st_instc_dir = "./data/std_inst_files/sqlite3"
    st_instc_dir = "./corpus/labled_instr"
    # basic_path = "/home/ubuntu/Desktop/Bert/add_token/corpus_new"
    loss_token_dir = "./data/loss_token"
    # add_vocal = "./corpus_new/sqlite3/corpus.txt"
    add_vocal = "./corpus/new_cor/corpus.txt"
    add_vocal_list = list()

    for instc in os.listdir(st_instc_dir):

        st_instc_path = st_instc_dir + os.sep + instc

        with open(st_instc_path, 'r') as fp:
            read_tokens = fp.readlines()

            for block_insts in read_tokens:
                insts = block_insts.split(' ')

                for inst in insts:
                    if inst not in add_vocal_list:
                        add_vocal_list.append(inst)


                # for inst in insts:
                #     inst_list1 = inst.split('~')
                #     operand = inst_list1[0]
                #
                #     if operand == '\n':
                #         continue
                #
                #     operand = operand.replace('\n', '')
                #
                #     if len(inst_list1) == 1:
                #
                #         if operand not in all_token:
                #             # operand_token.append(operand)
                #             judge_operand(operand)
                #
                #         continue
                #     if operand == 'sync' or operand == 'UND':
                #         continue
                #
                #     operator_list = inst_list1[1].split('>')
                #
                #     # judge_operand(operand)
                #
                #     if operand not in all_token:
                #         # operand_token.append(operand)
                #         judge_operand(operand)
                #
                #     for operatores in operator_list:
                #         if operatores == ' ' or operatores == '':
                #             break
                #         op_type = operatores.split('=')[0]
                #         operator = operatores.split('=')[1]
                #
                #         operator = operator.replace("\n","")
                #
                #         if int(op_type) == 1 or int(op_type) == 8 or int(op_type) == 9 or int(op_type) == 11:
                #             if operator not in operator_dict[1]:
                #                 operator_dict[1].append(operator)
                #
                #         elif int(op_type) == 2 or int(op_type) == 3 or int(op_type) == 4 or int(op_type) == 7:
                #             if operator not in operator_dict[3]:
                #                 operator_dict[3].append(operator)
                #
                #         elif int(op_type) == 5:
                #             if operator not in operator_dict[2]:
                #                 operator_dict[2].append(operator)
                #
                #         else:
                #             if operator not in operator_dict[5]:
                #                 operator_dict[4].append(operator)
                #

    # print (operand_token)

    # print (operator_dict[1])
    # print (operator_dict[2])
    # print (operator_dict[3])
    # print (operator_dict[4])
    # print (operator_dict[5])

    # with open(loss_token_dir + os.sep + "gsl.txt", 'w') as lp:
    #
    #     for lt in operand_token:
    #         lp.write(lt + '\n')

    with open(add_vocal,"w") as ap:

        for add_vocal_l in add_vocal_list:
            ap.write(add_vocal_l + "\n")

        # for reg_operator in REG_list:
        #     ap.write(reg_operator + "\n")
        #
        # for imm_operator in IMM_list:
        #     ap.write(imm_operator + "\n")
        #
        # for mem_operator in MEM_list:
        #     ap.write(mem_operator + "\n")
        #
        # for near_operator in NEAR_list:
        #     ap.write(near_operator + "\n")
        #
        # for data_operand in DataTransferInstr:
        #     ap.write(data_operand + "\n")
        #
        # for arith_operand in ArithmeticInstr:
        #     ap.write(arith_operand + "\n")
        #
        # for log_operand in LogicalInstr:
        #     ap.write(log_operand + "\n")
        #
        # for control_operand in ControlFlowInstr:
        #     ap.write(control_operand + "\n")
        #
        # for bit_operand in BitInstr:
        #     ap.write(bit_operand + "\n")
        #
        # for cond_trans_operand in ConditionalTransferInstr:
        #     ap.write(cond_trans_operand + "\n")
        #
        # for cond_set_operand in ConditionalSettingInstr:
        #     ap.write(cond_set_operand + "\n")
        #
        # for stack_operand in StackInstr:
        #     ap.write(stack_operand + "\n")
        #
        # for data_conv_operand in DataConversionInstr:
        #     ap.write(data_conv_operand + "\n")
        #
        # for compare_operand in CompareInstr:
        #     ap.write(compare_operand + "\n")

    # print ("123")

