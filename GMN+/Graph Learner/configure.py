import os

def get_use_features(features_type):
    """Do not use features if the option is selected."""
    if features_type == "nofeatures":
        return False
    return True


def get_bb_features_size(features_type):
    """Return features size by type."""
    if features_type == "nofeatures":
        return 7
    if features_type == "opc":
        return 768
    raise ValueError("Invalid features_type")


def update_config_datasetone(config_dict):
    """Config for Dataset-1."""

    experiment_type = 0

    basic_path = "../dataset"

    experiment_list = ["comparative", "ablation", "vulsearch"]

    inputdir = basic_path + os.sep + experiment_list[experiment_type]

    program_name = "all"
    # Training

    config_dict['training']['df_train_path'] = \
        os.path.join(inputdir, program_name, "train/train_func.csv")

    feature_dataset_basic_path = "/home/ubuntu/Desktop/2t_disk/gmm_data"
    # feature_dataset_basic_path = "/media/ubuntu/6a197cd7-03ba-495b-9b4a-942d0862fe17/gmm_data"
    config_dict['training']['features_train_path'] = feature_dataset_basic_path + os.sep + program_name + os.sep + "data_final/func_date2.json"
    # config_dict['training']['features_train_path'] = "/home/ubuntu/Desktop/2t_dist/gmm_data/glpk/data_final/train_data.json"
    # config_dict['training']['features_train_path'] = "/home/ubuntu/Desktop/2t_disk/gmm_data/glpk/data_nnbert_bentch/train_data.json"
    # config_dict['training']['features_train_path'] = "/home/ubuntu/Desktop/GMM/binary_function_similarity-main/oo_bert/oribert/sqlite3/data/train_data.json"
    # config_dict['training']['features_train_path'] = "/home/ubuntu/Desktop/2t_disk/gmm_data/sqlite3/data_nnbert/train_data.json"
    # config_dict['training']['features_train_path'] = "/home/ubuntu/Desktop/2t_disk/gmm_data/xml_2.9.2/data/train_data.json"
    # config_dict['training']['features_train_path'] = "/home/ubuntu/Desktop/2t_disk/gmm_data/xml_v/data_func/train_data.json"
    # config_dict['training']['features_train_path'] = "/home/ubuntu/Desktop/2t_disk/gmm_data/curl_v/data_func/train_data.json"
    # config_dict['training']['features_train_path'] = "/home/ubuntu/Desktop/2t_disk/gmm_data/xml_2.9.2/data_time/train_data.json"

    # Validation
    # valdir = os.path.join(inputdir, "pairs", "validation")
    # config_dict['validation'] = dict(
    #     positive_path=os.path.join(valdir, "pos_validation_Dataset-1.csv"),
    #     negative_path=os.path.join(valdir, "neg_validation_Dataset-1.csv"),
    #     features_validation_path=os.path.join(
    #         featuresdir,
    #         "Dataset-1_validation",
    #         "graph_func_dict_opc_200.json")
    # )

    all_compile_type = "arm_mips_x86_O0_O1_O2_O3"

    config_dict['validation'] = dict(
        positive_path=os.path.join(inputdir, program_name, "test", all_compile_type, "test_pos.csv"),
        negative_path=os.path.join(inputdir, program_name, "test", all_compile_type, "test_neg.csv"),
        features_validation_path=config_dict['training']['features_train_path']
    )

    config_dict['model_dir'] = inputdir + os.sep + program_name + os.sep + "model" + os.sep + "model.pth"

    config_dict['model_pp'] = inputdir + os.sep + program_name + os.sep + "model"

    config_dict['testing'] = dict(
        comparative_pair_inputs = os.path.join(basic_path, experiment_list[experiment_type], program_name, "test"),
        comparative_pair_outputs = os.path.join(basic_path, experiment_list[experiment_type], program_name, "result"),
        comparative_list_inputs=os.path.join(basic_path, experiment_list[experiment_type], program_name, "vul_list2"),
        comparative_list_outputs=os.path.join(basic_path, experiment_list[experiment_type], program_name, "result"),

        functions_list_inputs = os.path.join(inputdir, program_name, "test", all_compile_type, "test_pos.csv"),
        functions_list_outputs= os.path.join(inputdir, program_name, "test", all_compile_type, "test_pos.csv"),

        features_testing_path=config_dict['training']['features_train_path']
    )







def get_default_config():
    """The default configs."""
    # model_type = 'embedding'
    model_type = 'matching'
    features_type = "opc"
    # features_type = "nofeaturess"
    batch_size = 128
    # Set to `embedding` to use the graph embedding net.
    node_state_dim = 256
    edge_state_dim = 256
    graph_rep_dim = 128
    epoch_num = 12
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_state_dim=edge_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],
        n_prop_layers=3,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here. gru
        node_update_type='gru',
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different = True,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm= True,
        layers_accumulate = False,
        # set to `embedding` to use the graph embedding net.
        prop_type=model_type)
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config['similarity'] = 'cosine'  # other: euclidean, cosine

    config_dict = dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            edge_hidden_sizes=[edge_state_dim]),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type='sum'),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        model_type=model_type,

        features_type=features_type,
        batch_size=batch_size,
        bb_features_size = get_bb_features_size(features_type),

        data=dict(
            problem='graph_edit_distance',
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                n_nodes_range=[20, 20],
                p_edge_range=[0.2, 0.2],
                n_changes_positive=1,
                n_changes_negative=2,
                validation_dataset_size=1000),
            use_features = get_use_features(features_type)),
        training=dict(
            batch_size=batch_size,
            epoch_num = epoch_num,
            learning_rate=1e-3,
            mode='triplet',
            # mode='pair',
            loss='margin',  # other: hamming
            margin=0.2,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            n_training_steps=3000,
            # Print training information every this many training steps.
            print_after=10,
            # Evaluate on validation set every `eval_after * print_after` steps.
            eval_after=10),
        evaluation=dict(
            batch_size=batch_size,
            isvalid = True),
        seed=8,
    )

    update_config_datasetone(config_dict)

    return config_dict




