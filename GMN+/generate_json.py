import json
import os
import numpy as np
import config
from collections import defaultdict

final_json_path = config.STEP3_SAVE_DIR

jsonset_path = config.STEP3_JSON_DIR

vuljson_path = config.STEP3_DATASET_DIR

vul_npyset_dir = config.STEP3_VUL_EMBEDDING_DIR + os.sep + config.STEP2_SOFTWARE_NAME

dataset_npyset_dir = config.STEP3_DATASET_EMBEDDING_DIR

feature_size = 768

def create_graph(succs,n_num):

    row_list = list()
    col_list = list()
    data_list = list()


    row_str = ""
    col_str = ""
    data_str = ""
    # for i in range(len(succs)):
    #     end_list = succs[i]
    #     for j in range(len(end_list)):
    #         row_list.append(i)
    #         row_str += str(i) + ';'
    #
    #         col_list.append(end_list[j])
    #         col_str += str(end_list[j]) + ';'
    #
    #         data_list.append(1)
    #         data_str += str(1) + ';'

    for ee in succs:
        if len(ee)==2:
            row_list.append(ee[0])
            row_str += str(ee[0]) + ';'

            col_list.append(ee[1])
            col_str += str(ee[1]) + ';'

            data_list.append(1)
            data_str += str(1) + ';'

    row_str = row_str[:-1]
    col_str = col_str[:-1]
    data_str = data_str[:-1]

    graph_str = row_str + "::" + col_str + "::" + data_str + "::" + str(n_num) + "::" + str(n_num)

    return graph_str

def create_features(node_features, n_num):

    row_list = list()
    col_list = list()
    data_list = list()

    row_str = ""
    col_str = ""
    data_str = ""
    for i in range(len(node_features)):

        for j in range(len(node_features[i])):

            row_list.append(i)
            row_str += str(i) + ';'

            col_list.append(j)
            col_str += str(j) + ';'

            data_list.append(node_features[i][j])
            data_str += str(node_features[i][j]) + ';'


    row_str = row_str[:-1]
    col_str = col_str[:-1]
    data_str = data_str[:-1]

    feature_sizes = len(node_features[0])
    feature_str = row_str + "::" + col_str + "::" + data_str + "::" + str(n_num) + "::" + str(feature_size)

    return feature_str



def create_features_with_embedding(n_num, json_path, func_name, npyset_dir):

    row_list = list()
    col_list = list()
    data_list = list()

    row_str = ""
    col_str = ""
    data_str = ""

    json_path_list = json_path.split(os.sep)

    json_name = json_path_list[-1]

    npy_path = npyset_dir + os.sep + json_name.split(".json")[0] + os.sep + func_name + ".npy"

    instEmbedding = np.load(npy_path, allow_pickle=True)

    for i in range(instEmbedding.shape[0]):

        for j in range(instEmbedding[i].shape[0]):

            row_list.append(i)
            row_str += str(i) + ';'

            col_list.append(j)
            col_str += str(j) + ';'

            data_list.append(instEmbedding[i][j])
            data_str += str(instEmbedding[i][j]) + ';'


    row_str = row_str[:-1]
    col_str = col_str[:-1]
    data_str = data_str[:-1]

    feature_str = row_str + "::" + col_str + "::" + data_str + "::" + str(n_num) + "::" + str(feature_size)

    return feature_str





if __name__ == '__main__':

    funcname_list = list()

    functions_dict = defaultdict(dict)

    for json_file in os.listdir(jsonset_path):
            json_path = jsonset_path + os.sep + json_file
            print (json_path)
            with open(json_path) as js:
                for line in js:
                    g_info = json.loads(line.strip())
                    flag = 0

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

                    ida_path = g_info["src"]
                    func_name = g_info["fname"]
                    bb_num = str(g_info["n_num"])
                    arch = json_file.split('_')[2]
                    bit = "64"
                    compiler = "gcc"
                    version = json_file.split('_')[1]
                    optimizations = json_file.split('_')[3]
                    succs = g_info["succs"]

                    a_o = arch + '_' + optimizations
                    features = g_info["features"]
                    feature_list = list()

                    for feature in features:
                        feature_list.append(feature[:feature_size])

                    # ee = create_features(feature_list, bb_num)
                    # graph_str = create_graph(succs,bb_num)
                    #
                    # feature_str = create_features_with_embedding(bb_num, json_path, func_name)
                    fnn = func_name
                    functions_dict[fnn][a_o] = {
                        'graph': create_graph(succs,bb_num),
                        # 'opc': create_features(feature_list, bb_num)
                        'opc': create_features_with_embedding(bb_num, json_path, func_name, vul_npyset_dir)
                    }

    for json_file in os.listdir(vuljson_path):
            json_path = vuljson_path + os.sep + json_file
            print (json_path)
            with open(json_path) as js:
                for line in js:
                    g_info = json.loads(line.strip())
                    flag = 0

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

                    ida_path = g_info["src"]
                    func_name = g_info["fname"]
                    bb_num = str(g_info["n_num"])
                    arch = json_file.split('_')[2]
                    bit = "64"
                    compiler = "gcc"
                    version = json_file.split('_')[1]
                    optimizations = json_file.split('_')[3]
                    succs = g_info["succs"]

                    a_o = arch + '_' + optimizations
                    features = g_info["features"]
                    feature_list = list()

                    for feature in features:
                        feature_list.append(feature[:feature_size])

                    # ee = create_features(feature_list, bb_num)
                    # graph_str = create_graph(succs,bb_num)
                    #
                    # feature_str = create_features_with_embedding(bb_num, json_path, func_name)
                    fnn = func_name
                    functions_dict[fnn][a_o] = {
                        'graph': create_graph(succs,bb_num),
                        # 'opc': create_features(feature_list, bb_num)
                        'opc': create_features_with_embedding(bb_num, json_path, func_name, dataset_npyset_dir)
                    }


    if not os.path.exists(final_json_path):
        os.makedirs(final_json_path)

    with open(final_json_path + os.sep + "func_date.json" , 'w') as f_out:
        json.dump(functions_dict, f_out)