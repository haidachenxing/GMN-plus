import json
import os
import random
import numpy as np
import config

target_jsonset_path = config.STEP3_JSON_DIR
dataset_jsonset_path = config.VUL_JSONSET_PATH
csv_path = config.STEP3_FUNC_LIST_DIR

def get_funcdict(jsonset_path):
    fname_num = 0
    fname_dict = {}
    fname_list = []

    for f_name in os.listdir(jsonset_path):
        with open(jsonset_path + os.sep + f_name) as inf:

            for line in inf:
                g_info = json.loads(line.strip())
                flag = 0

                if g_info['n_num'] < 5:
                    continue

                if g_info['n_num'] >= 500:
                    continue

                insts = g_info['features']

                for inst in insts:

                    if len(inst[11]) == 0:
                        flag = 1
                        break
                    # if len(inst[12]) == 0:
                    #     flag = 1
                    #     break
                if flag:
                    continue

                if g_info['fname'] not in fname_dict:
                    fname_dict[g_info['fname']] = []
                    fname_list.append(g_info['fname'])
                    fname_dict[g_info['fname']].append(f_name)
                    fname_num += 1
                else:
                    fname_dict[g_info['fname']].append(f_name)

    return fname_dict,fname_list

def newgenerate_vul(dataset_func_dict, dataset_func_list, target_func_dict, target_func_list):

    target_funcpair_set = list()

    for target_func in target_func_list:
        target_funcpair_list = list()
        for target_func_source in target_func_dict[target_func]:

            for dataset_func in dataset_func_list:
                for dataset_func_source in dataset_func_dict[dataset_func]:
                    funcpair_str = target_func_source + ',' + target_func_source.split('_')[2] \
                                    + ',' + target_func_source.split('_')[3] + ',' + target_func \
                                   + ',' + dataset_func_source + ',' + dataset_func_source.split('_')[2]\
                                    + ',' + dataset_func_source.split('_')[3] + ',' + dataset_func + ",-1\n"

                    target_funcpair_list.append(funcpair_str)

        target_funcpair_set.append(target_funcpair_list)

    return target_funcpair_set



if __name__ == '__main__':


    dataset_func_dict, dataset_func_list = get_funcdict(dataset_jsonset_path)
    target_func_dict, target_func_list = get_funcdict(target_jsonset_path)

    target_funcpair_set = newgenerate_vul(dataset_func_dict, dataset_func_list, target_func_dict, target_func_list)

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for i in range(len(target_func_list)):
        func_name = target_func_list[i]
        target_funcpair = target_funcpair_set[i]

        with open(csv_path + os.sep + func_name + "_tar.csv", "w") as np:

            flag = 0
            tet = "ida_path_1" + ',' + "arch_1" + ',' + "opti_1" + ',' + "func_name_1" + ',' + "ida_path_2" + ',' + "arch_2" + ',' + "opti_2" + ',' + "func_name_2" + ',' + "type" + "\n"
            np.write(tet)
            for funcpair_str in target_funcpair:
                np.write(funcpair_str)
                flag += 1

