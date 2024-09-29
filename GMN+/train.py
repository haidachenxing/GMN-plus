import torch

from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_label_loss, new_triplet_loss,new_triplet_local_loss
from utils import *
from configure import *
import numpy as np
import torch.nn as nn
import collections
import time
import os
import pandas as pd
import gc

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
# torch.cuda.set_per_process_memory_fraction(device=device, fraction=0.8)

type = "train"

test_type_list = [
                    [["arm", "mips"], ["O1"]], [["arm", "x86"], ["O1"]], [["mips", "x86"], ["O1"]], [["arm", "mips", "x86"], ["O1"]],
                    [["arm"], ["O1","O2"]], [["arm"], ["O0", "O3"]], [["arm"], ["O0", "O1", "O2", "O3"]],
                    [["arm", "mips", "x86"], ["O0", "O1", "O2", "O3"]]
                  ]

eval_type = 1
# device = torch.device('cpu')
# Print configure
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))


model_path = config['model_dir']

# basic_path = "/home/ubuntu/Desktop/gmm/Graph-Matching-Networks/GMN/train_dataset/dataset_oobert/sqlite3_arch/result"
# auc_path =  basic_path + os.sep + "auc.csv"
# result_path = basic_path + os.sep + "siml.csv"
# loss_path = basic_path + os.sep + "loss.csv"

print (model_path)
print (config['training'])

# Set random seeds
seed = config['seed']
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# training_set, validation_set = new_build_datasets(config)

training_set = new_build_train_dattasets(config)
# validation_set = new_build_testing_datasets(config)
# validation_set = list()
if config['training']['mode'] == 'pair':
    training_data_iter = training_set.pairs()
    first_batch_graphs, _ = next(training_data_iter)
else:
    training_data_iter = training_set.triplets()
    first_batch_graphs = next(training_data_iter)

# node_feature_dim = 768
# edge_feature_dim = 1536

node_feature_dim = first_batch_graphs.node_features.shape[-1]
edge_feature_dim = first_batch_graphs.edge_features.shape[-1]
print("node_feature_dim:" + str(node_feature_dim))
print("edge_feature_dim:" + str(edge_feature_dim))

model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

accumulated_metrics = collections.defaultdict(list)

training_n_graphs_in_batch = config['training']['batch_size']
if config['training']['mode'] == 'pair':
    training_n_graphs_in_batch *= 2
elif config['training']['mode'] == 'triplet':
    training_n_graphs_in_batch *= 2
else:
    raise ValueError('Unknown training mode: %s' % config['training']['mode'])

if not os.path.exists(model_path):
    os.mknod(model_path)

if os.path.exists(model_path):

    if type == "train":

        for epoch in range(config['training']['epoch_num']):
            model.train(mode=True)

            training_data_iter = training_set.triplets()

            for i_iter in range(config['training']['n_training_steps']):
                t_start = time.time()

                try:
                    batch = next(training_data_iter)
                except:
                    print("StopIteration")
                    continue

                node_features, edge_features, from_idx, to_idx, graph_idx, from_idx_list, to_idx_list, graph_ad, edge_idx = get_graph(
                    batch)

                graph_vectors, node_states = model(node_features.to(device), edge_features.to(device),
                                                from_idx.to(device),
                                                to_idx.to(device),
                                                graph_idx.to(device), training_n_graphs_in_batch,
                                                from_idx_list.to(device), to_idx_list.to(device),
                                                graph_ad.to(device), edge_idx.to(device))

                x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
                loss = new_triplet_loss(x_1, y, x_2, z,
                                        loss_type=config['training']['loss'],
                                        margin=config['training']['margin'])

                sim_pos = torch.mean(compute_similarity(config, x_1, y))
                sim_neg = torch.mean(compute_similarity(config, x_2, z))


                graph_vec_scale = torch.mean(graph_vectors ** 2)
                if config['training']['graph_vec_regularizer_weight'] > 0:
                    loss += (config['training']['graph_vec_regularizer_weight'] *
                             0.5 * graph_vec_scale)

                optimizer.zero_grad()
                loss.backward(torch.ones_like(loss))  #
                nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_value'])
                optimizer.step()

                sim_diff = sim_pos - sim_neg
                accumulated_metrics['loss'].append(loss)
                accumulated_metrics['sim_pos'].append(sim_pos)
                accumulated_metrics['sim_neg'].append(sim_neg)
                accumulated_metrics['sim_diff'].append(sim_diff)

                # evaluation
                if (i_iter + 1) % config['training']['print_after'] == 0:
                    metrics_to_print = {
                        k: torch.mean(v[0]) for k, v in accumulated_metrics.items()}
                    info_str = ', '.join(
                        ['%s %.4f' % (k, v) for k, v in metrics_to_print.items()])
                    # reset the metrics
                    accumulated_metrics = collections.defaultdict(list)

                    # if ((i_iter + 1) // config['training']['print_after'] %
                    #         config['training']['eval_after'] == 0):
                    print('iter %d, %s, time %.2fs' % (
                        i_iter + 1, info_str, time.time() - t_start))
                    t_start = time.time()


            # if eval_type:
            #     model.eval()
            #     with torch.no_grad():
            #         accumulated_pair_auc = []
            #         for batch in validation_set.pairs():
            #             node_features, edge_features, from_idx, to_idx, graph_idx, from_idx_list, to_idx_list, graph_ad, edge_idx, labels = get_graph(
            #                 batch)
            #
            #             labels = labels.to(device)
            #
            #             eval_pairs, node_status = model(node_features.to(device), edge_features.to(device),
            #                                             from_idx.to(device),
            #                                             to_idx.to(device),
            #                                             graph_idx.to(device),
            #                                             config['evaluation']['batch_size'] * 2,
            #                                             from_idx_list.to(device), to_idx_list.to(device),
            #                                             graph_ad.to(device), edge_idx.to(device))
            #
            #             x, y = reshape_and_split_tensor(eval_pairs, 2)
            #             similarity = compute_similarity(config, x, y)
            #             pair_auc = auc(similarity, labels)
            #             accumulated_pair_auc.append(pair_auc)
            #
            #         eval_metrics = {
            #             'pair_auc': np.mean(accumulated_pair_auc),
            #             # 'triplet_acc': np.mean(accumulated_triplet_acc)
            #         }
            #         info_str += ', ' + ', '.join(
            #             ['%s %.4f' % ('val/' + k, v) for k, v in eval_metrics.items()])
            #
            #         print(eval_metrics)
            #         print(epoch)
            #     model.train()

            gc.collect()
            torch.cuda.empty_cache()

            model_epoch_name = "model_" + str(epoch) + ".pth"

            if not os.path.exists(config['model_pp'] + os.sep + model_epoch_name):
                os.mknod(config['model_pp'] + os.sep + model_epoch_name)

            torch.save(model, config['model_pp'] + os.sep + model_epoch_name)


        torch.save(model, model_path)

    if type == "test_func":

        model = torch.load(config['model_dir'])

        inputs_list = os.listdir(config['testing']['functions_list_inputs'])

        for func_name in inputs_list:
            print(func_name)
            df_input_path = config['testing']['functions_list_inputs'] + os.sep + func_name
            df_output_path = config['testing']['functions_list_outputs'] + os.sep + func_name

            df = pd.read_csv(df_input_path, index_col=0)

            batch_generator = build_testing_generator(
                config,
                df_input_path)

            similarity_list = list()
            model.eval()
            with torch.no_grad():
                accumulated_pair_auc = []
                accumulated_pair_sim = []
                accumulated_pair_label = []

                for batch in batch_generator.pairs():

                    node_features, edge_features, from_idx, to_idx, graph_idx, from_idx_list, to_idx_list, graph_ad, edge_idx = get_graph(
                        batch)

                    # eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                    #                    to_idx.to(device),
                    #                    graph_idx.to(device), config['evaluation']['batch_size'] * 2)

                    eval_pairs, node_status = model(node_features.to(device), edge_features.to(device),
                                                    from_idx.to(device),
                                                    to_idx.to(device),
                                                    graph_idx.to(device), config['evaluation']['batch_size'] * 2,
                                                    from_idx_list.to(device), to_idx_list.to(device),
                                                    graph_ad.to(device), edge_idx.to(device))

                    x, y = reshape_and_split_tensor(eval_pairs, 2)
                    similarity = compute_similarity(config, x, y)
                    # pair_auc = auc(similarity, labels)
                    # accumulated_pair_auc.append(pair_auc)

                    simls = similarity.cpu().detach().numpy().tolist()
                    for siml in simls:
                        accumulated_pair_sim.append(siml)

            df['sim'] = accumulated_pair_sim[:df.shape[0]]

            df.to_csv(df_output_path)
            print("Result CSV saved to {}".format(df_output_path))

            gc.collect()
            torch.cuda.empty_cache()

