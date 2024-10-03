import numpy as np
import torch


def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)


def approximate_hamming_similarity(x, y):
    """Approximate Hamming similarity."""
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)


def new_pair_loss(is_pos, is_neg, sim):
    """Compute pairwise loss.

    Args:
      x: [N, D] float tensor, representations for N examples.
      y: [N, D] float tensor, representations for another N examples.
      labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
        and y[i] are similar, and -1 otherwise.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.

    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.
    """



    # pos_sim_list = sim[is_pos.nonzero(as_tuple=True)]
    # neg_sim_list = sim[is_neg.nonzero(as_tuple=True)]

    n_pos = torch.sum(is_pos)
    n_neg = torch.sum(is_neg)

    sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8)
    sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8)

    loss = torch.exp(sim_neg - sim_pos)

    return loss





def pairwise_loss(x, y, labels, loss_type='margin', margin=0.5):
    """Compute pairwise loss.

    Args:
      x: [N, D] float tensor, representations for N examples.
      y: [N, D] float tensor, representations for another N examples.
      labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
        and y[i] are similar, and -1 otherwise.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.

    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.
    """

    labels = labels.float()

    if loss_type == 'margin':
        return torch.relu(margin - labels * (1 - euclidean_distance(x, y)))
    elif loss_type == 'hamming':
        return 0.25 * (labels - approximate_hamming_similarity(x, y)) ** 2
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)



def new_triplet_loss(x_1, y, x_2, z, loss_type='margin', margin=0.2):
    """Compute triplet loss.

    This function computes loss on a triplet of inputs (x, y, z).  A similarity or
    distance value is computed for each pair of (x, y) and (x, z).  Since the
    representations for x can be different in the two pairs (like our matching
    model) we distinguish the two x representations by x_1 and x_2.

    Args:
      x_1: [N, D] float tensor.
      y: [N, D] float tensor.
      x_2: [N, D] float tensor.
      z: [N, D] float tensor.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.

    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.
    """
    if loss_type == 'margin':

        loss1 = torch.log(1 + torch.exp(torch.nn.functional.cosine_similarity(x_2, z) - torch.nn.functional.cosine_similarity(x_1, y)))
        # loss2 =
        return loss1

        # text =
        # return torch.log( 1 + torch.exp(margin +
        #                   euclidean_distance(x_1, y) -
        #                   euclidean_distance(x_2, z)))
    elif loss_type == 'hamming':
        return 0.125 * ((approximate_hamming_similarity(x_1, y) - 1) ** 2 +
                        (approximate_hamming_similarity(x_2, z) + 1) ** 2)
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)




def compute_cross_attention(x, y):

    cx = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), torch.tensor(1e-12))))
    cy = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), torch.tensor(1e-12))))
    a = torch.mm(cx, torch.transpose(cy, 1, 0))

    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), torch.tensor(1e-12))))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), torch.tensor(1e-12))))
    a = torch.mm(x, torch.transpose(y, 1, 0))

    a_x = torch.softmax(a, dim=1)  # i->j

    attention_x = torch.mm(a_x, y)

    return attention_x



def batch_block_pair_attention(data,
                               block_idx,
                               n_blocks,):

    results = []

    pos_results = []
    neg_results = []

    # This is probably better than doing boolean_mask for each i
    partitions = []
    for i in range(n_blocks):
        partitions.append(data[block_idx == i, :])

    flag = 0   #pos neg
    for i in range(0, n_blocks, 2):

        x = partitions[i]
        y = partitions[i + 1]
        attention_x, attention_y = compute_cross_attention(x, y)

        if flag % 2 == 0:
            pos_results.append(x - attention_x)

        else:
            neg_results.append(x - attention_x)

        flag += 1
        # x = partitions[i]
        # y = partitions[i + 1]
        # attention_x, attention_y = compute_cross_attention(x, y)
        # pos_results.append(x - attention_x)
        # neg_results.append(y - attention_y)

        # results.append(attention_x)
        # results.append(attention_y)
    # results = torch.cat(results, dim=0)

    # pos_results = torch.cat(pos_results, dim=0)
    # neg_results = torch.cat(neg_results, dim=0)

    return pos_results, neg_results



    results = []

    # This is probably better than doing boolean_mask for each i
    partitions = []
    for i in range(n_blocks):
        partitions.append(data[block_idx == i, :])

    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        attention_x = compute_cross_attention(x, y)

        torch.nn.functional.cosine_similarity(x,attention_x)
        result = torch.sum((x - attention_x),dim=0) / len(x)
        results.append(result)

    results = torch.cat(results, dim=0)


    return results


def triplet_label_loss(pos_similarity, neg_similarity):
    pos_label = torch.full(pos_similarity.shape, fill_value=1, dtype=torch.float32, device=torch.device('cuda:0'))
    neg_label = torch.full(neg_similarity.shape, fill_value=-1, dtype=torch.float32, device=torch.device('cuda:0'))

    label_loss = pos_label - pos_similarity + neg_label - neg_similarity

    return label_loss

def new_triplet_local_loss(node_states,graph_idx, n_graphs, batch_size, node_size):


    diff_pos, diff_neg = batch_block_pair_attention(
        node_states, graph_idx, n_graphs)

    # diff_pos, diff_neg = node_states - cross_graph_attention

    cross_loss = torch.reshape(diff_pos,[batch_size, node_size])

    results = []

    for i in range(0, batch_size, 2):
        t1 = torch.abs(cross_loss[i])
        t2 = torch.abs(cross_loss[i+1])

        # l1 = torch.sum(t1,dim=0)
        # l2 = torch.sum(t2,dim=0)

        # result3 = torch.exp(l1 - l2)
        result = torch.log(1 + torch.nn.functional.cosine_similarity(t1, t2, dim=0))
        results.append(result)


    return torch.tensor(results,device=torch.device('cuda:0'))






def triplet_loss(x_1, y, x_2, z, loss_type='margin', margin=0.2):
    """Compute triplet loss.

    This function computes loss on a triplet of inputs (x, y, z).  A similarity or
    distance value is computed for each pair of (x, y) and (x, z).  Since the
    representations for x can be different in the two pairs (like our matching
    model) we distinguish the two x representations by x_1 and x_2.

    Args:
      x_1: [N, D] float tensor.
      y: [N, D] float tensor.
      x_2: [N, D] float tensor.
      z: [N, D] float tensor.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.

    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.
    """
    if loss_type == 'margin':
        return torch.relu(margin +
                          euclidean_distance(x_1, y) -
                          euclidean_distance(x_2, z))

        # text =
        # return torch.log( 1 + torch.exp(margin +
        #                   euclidean_distance(x_1, y) -
        #                   euclidean_distance(x_2, z)))
    elif loss_type == 'hamming':
        return 0.125 * ((approximate_hamming_similarity(x_1, y) - 1) ** 2 +
                        (approximate_hamming_similarity(x_2, z) + 1) ** 2)
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)
