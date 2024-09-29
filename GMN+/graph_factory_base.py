import json
import math
import networkx as nx
import pandas as pd

from graph_factory_utils import *
from tqdm import tqdm


import collections

# GraphData = collections.namedtuple('GraphData', [
#     'from_idx',
#     'to_idx',
#     'node_features',
#     'edge_features',
#     'graph_idx',
#     'n_graphs'])


class GraphFactoryBase(object):
    """Base class for all the graph similarity learning datasets.

    This class defines some common interfaces a graph similarity dataset can have,
    in particular the functions that creates iterators over pairs and triplets.
    """

    def triplets(self):
        """Create an iterator over triplets.

        Note:
          batch_size: int, number of triplets in a batch.

        Yields:
          graphs: a `GraphData` instance.  The batch of triplets put together.  Each
            triplet has 3 graphs (x, y, z).  Here the first graph is duplicated once
            so the graphs for each triplet are ordered as (x, y, x, z) in the batch.
            The batch contains `batch_size` number of triplets, hence `4*batch_size`
            many graphs.
        """
        pass

    def pairs(self):
        """Create an iterator over pairs.

        Note:
          batch_size: int, number of pairs in a batch.

        Yields:
          graphs: a `GraphData` instance.  The batch of pairs put together.  Each
            pair has 2 graphs (x, y).  The batch contains `batch_size` number of
            pairs, hence `2*batch_size` many graphs.
          labels: [batch_size] int labels for each pair, +1 for similar, -1 for not.
        """
        pass



class GraphFactoryInference(GraphFactoryBase):

    def __init__(self, func_path, feat_path, batch_size,
                 use_features, features_type, bb_features_size):
        """
            Args:
                func_path: CSV file with function pairs
                feat_path: json file with function features
                batch_size: size of the batch for each iteration
                use_features: if True, load the graph node features
                features_type: used to select the appropriate decoder and data
                bb_features_size: numer of features at the basic-block level
        """
        if batch_size <= 0 or batch_size % 2 != 0:
            raise SystemError("Batch size must be even and >= 0")

        self._batch_size = batch_size
        log.info("Batch size (inference): {}".format(self._batch_size))

        self._use_features = use_features
        self._features_type = features_type
        self._bb_features_size = bb_features_size
        # self._decoder = str_to_scipy_sparse
        self._decoder = str_to_scipy_sparse_features

        # Load function pairs
        log.debug("Reading {}".format(func_path))
        self._func = pd.read_csv(func_path)

        # Load function features
        log.debug("Loading {}".format(feat_path))
        with open(feat_path) as gfd_in:
            self._fdict = json.load(gfd_in)

        # Initialize the iterator
        self._get_next_pair_it = self._get_next_pair()

        # Number of function pairs
        self._num_func_pairs = self._func.shape[0]
        log.info("Num func pairs (inference): {}".format(self._num_func_pairs))

        # _get_next_pair() returns a pair of functions
        # _batch_size must be even and >= 2
        # _num_batches is the number of iterations to cover the input data
        # Example:
        #   * 100 functions. Batch_size = 20; 5 iterations
        #   * 100 functions. Batch_size = 16; 7 iterations
        self._num_batches = math.ceil(
            self._num_func_pairs / self._batch_size)
        log.info("Num batches (inference): {}".format(self._num_batches))

    def _get_next_pair(self):
        """The function implements an infinite loop over the input data."""
        while True:
            log.info("(Re-)initializing the iterators")
            # (Re-)initialize the iterators
            iterator = self._func.iterrows()

            for _ in range(self._num_func_pairs):
                # Get the next row
                r = next(iterator)[1]
                # Get the features for the left function
                f_aol = r['arch_1'] + '_' + r['opti_1']
                f_aor = r['arch_2'] + '_' + r['opti_2']

                f_l = self._fdict[r['func_name_1']][f_aol]
                # ... and for the right one
                f_r = self._fdict[r['func_name_2']][f_aor]
                # f_l = self._fdict[r['idb_path_1']][r['fva_1']]
                # # ... and for the right one
                # f_r = self._fdict[r['idb_path_2']][r['fva_2']]

                if self._use_features:
                    yield (
                        (
                            nx.DiGraph(str_to_scipy_sparse(f_l['graph'])),
                            nx.DiGraph(str_to_scipy_sparse(f_r['graph']))
                        ),
                        (
                            self._decoder(f_l[self._features_type]),
                            self._decoder(f_r[self._features_type])
                        )
                    )
                else:
                    yield (
                        (
                            nx.DiGraph(str_to_scipy_sparse(f_l['graph'])),
                            nx.DiGraph(str_to_scipy_sparse(f_r['graph']))
                        ),
                        (
                            -1, -1
                        )
                    )

    def pairs(self):
        """Yields batches of pair data."""
        for _ in tqdm(range(self._num_batches),
                      total=self._num_batches):
            batch_graphs = list()
            batch_features = list()

            for _ in range(self._batch_size):
                g_pair, f_pair = next(self._get_next_pair_it)
                batch_graphs.append((g_pair[0], g_pair[1]))
                batch_features.append((f_pair[0], f_pair[1]))

            # Pack everything in a graph data structure
            packed_graphs = pack_batch(
                batch_graphs,
                batch_features,
                self._use_features,
                nofeatures_size=self._bb_features_size)

            yield packed_graphs

    def triplets(self):
        """ Yields batches of triplet data. For training only."""
        pass
