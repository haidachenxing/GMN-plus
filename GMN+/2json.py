# -*- coding: utf-8 -*-
import pickle
import json
import os
# from raw_graphs import *
import config
import networkx

dirpath = config.STEP1_IDAFILE_DIR
files = os.listdir(dirpath)

for fl in files:
    jsonpath = config.STEP1_JSON_DIR + os.sep + fl.split('.ida')[0] + '_features.json'  # output path
    filepath = os.path.join(dirpath, fl)
    print(filepath)
    with open(filepath, 'rb') as f:
        cfgs = pickle.load(f)

    if not os.path.exists(config.STEP1_JSON_DIR):
        os.makedirs(config.STEP1_JSON_DIR)
    with open(jsonpath, 'w') as jf:
        for cfg in cfgs.raw_graph_list:
            dic = {}
            dic['src'] = filepath
            dic['fname'] = cfg.funcname
            dic['n_num'] = len(cfg.g.node)
            suc = []
            aee = networkx.adjacency_matrix(cfg.g)
            for key in cfg.g.adj:
                dd = cfg.g.adj[key]
                suc.append(list(dd.keys()))
            # for n, nbrsdict in cfg.g.adjacency():
            #     suc.append(list(nbrsdict.keys()))
            dic['succs'] = suc
            dic['features'] = []
            for i in range(len(cfg.g.node)):
                fvec = []
                fvec.append(len(cfg.g.node[i]['v'][0]))  # 'consts'
                fvec.append(len(cfg.g.node[i]['v'][1]))  # 'strings'
                fvec.append(cfg.g.node[i]['v'][2])  # 'offs'
                fvec.append(cfg.g.node[i]['v'][3])  # 'numAs'
                fvec.append(cfg.g.node[i]['v'][4])  # 'numCalls'
                fvec.append(cfg.g.node[i]['v'][5])  # 'numIns'
                fvec.append(cfg.g.node[i]['v'][6])  # 'numLIs'
                fvec.append(cfg.g.node[i]['v'][7])  # 'numTIs'
                toStDis = cfg.g.node[i]['v'][8]
                if type(toStDis) == dict:
                    fvec.append(0)  # 'toStDis'
                else:
                    fvec.append(toStDis)  # 'toStDis'
                fvec.append(cfg.g.node[i]['v'][9])  # 'toEdDis'
                fvec.append(cfg.g.node[i]['v'][10])  # 'between'
                # fvec.append(cfg.g.node[i]['v'][11])  # insts
                fvec.append(cfg.g.node[i]['v'][12])  # prepro insts
                dic['features'].append(fvec)
            dic['func_feature'] = cfg.fun_features[:-2]

            data = json.dumps(dic)
            jf.write(data + '\n')

