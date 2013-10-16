import numpy as np
from scipy import sparse as sp
import pymaxflow
import time

def remove_rows_cols(M, mask):
    # We can write the M.data in place
    indptr, indices, data = M.indptr, M.indices, M.data
    assert isinstance(M, sp.csr_matrix)
    mask = mask.astype(np.bool)
    for row in range(M.shape[0]):
        if mask[row]:
            M.data[indptr[row]:indptr[row + 1]] = 0
    M.data[mask[indices]] = 0
    M.eliminate_zeros()

class SlimGraph(object):
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.source_caps = np.zeros(num_nodes)
        self.sink_caps = np.zeros(num_nodes)
        self.edge_caps = []

    def add_source_capacities(self, caps):
        self.source_caps += caps

    def add_sink_capacities(self, caps):
        self.sink_caps += caps

    def add_edge_capacities(self, i, j, caps, symmetric=False):
        self.edge_caps.append(sp.csr_matrix((caps, (i, j)), shape=(self.num_nodes, self.num_nodes)))
        if symmetric:
            self.edge_caps.append(sp.csr_matrix((caps, (j, i)), shape=(self.num_nodes, self.num_nodes)))

    def slim(self, slim=True):
        edge_caps = sum(self.edge_caps)

        if not slim:
            if edge_caps is 0:
                edge_caps = sp.csr_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            return np.zeros(self.num_nodes, dtype=np.bool), np.zeros(self.num_nodes, dtype=np.bool), edge_caps

        for idx in range(20):
            source_sink_flow = self.source_caps - self.sink_caps
            inflow_source = np.maximum(0, source_sink_flow)
            outflow_sink = np.maximum(0, -source_sink_flow)

            inflows = edge_caps.sum(axis=1).A.ravel()
            ecsc = edge_caps.tocsc()
            outflows = ecsc.sum(axis=0).A.ravel()
            merge_to_source = (inflow_source > outflows)
            merge_to_sink = (outflow_sink > inflows)
            if not (np.any(merge_to_source) or np.any(merge_to_sink)):
                print "none", inflow_source[:10], outflows[:10]
                return
            # for nodes linked to source, move node->neighbor capacity to source->neighbor

            self.source_caps += (edge_caps * merge_to_source.reshape((-1, 1))).ravel()
            self.sink_caps += (ecsc.T * merge_to_sink.reshape((-1, 1))).ravel()
            remove_rows_cols(edge_caps, merge_to_sink | merge_to_source)
            self.source_caps[merge_to_source] = 0
            self.sink_caps[merge_to_sink] = 0
            print "iter", idx, "merged", np.sum(merge_to_source) + np.sum(merge_to_sink), '/', self.num_nodes, "NNZ", edge_caps.nnz
        return merge_to_source, merge_to_sink, edge_caps

    def solve(self, slim=False, verbose=False):
        st = time.time()
        merge_source, merge_sink, caps = self.slim(slim)
        g = pymaxflow.PyGraph(self.num_nodes, caps.nnz)

        g.add_node(self.num_nodes)
        caps = caps.tocoo()
        row, col, data = caps.row, caps.col, caps.data
        g.add_edge_vectorized(row, col, data, 0 * data)
        self.source_caps[merge_source] = 1
        self.sink_caps[merge_sink] = 1
        g.add_tweights_vectorized(np.arange(self.num_nodes, dtype=np.int32),
                                  self.source_caps.astype(np.float32),
                                  self.sink_caps.astype(np.float32))
        st2 = time.time()
        g.maxflow()
        if verbose:
            print "Solve took {0} seconds, {1} in maxflow".format(time.time() - st, time.time() - st2)
        return g.what_segment_vectorized()
