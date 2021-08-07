import tensorflow as tf
import numpy as np
import scipy.sparse

def convert_scipy_sparse_to_sparse_tensor(matrix):
    """Converts a sparse matrix to a Tensorflow SparseTensor.

    Args:
    matrix: A scipy sparse matrix.

    Returns:
    A ternsorflow sparse matrix (rank-2 tensor).
    """
    matrix = matrix.tocoo()
    return tf.sparse.SparseTensor(
      np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)

def categorical(logits, num_samples, dtype=None, seed=None):
    """Takes a two-dimensional tensor of logits and returns sampled indices for each row.

    Args:
    logits: A two-dimensional tensor
    num_samples: The number of samples to take from each row
    dtype: data type of output
    seed: random seed for sampling

    Returns:
    A two-dimensional tensor containing the sampled indices for each row
    """
    logits = tf.convert_to_tensor(logits, name="logits")
    dt = tf.float32
    batch = logits.shape.dims[0].value
    num_classes = logits.shape.dims[1].value
    if num_classes == 0:
    # Delegates to native op to raise the proper error.
        return tf.random.categorical(
            logits, num_samples, output_dtype=dtype)
    # u ~ Uniform[0.0, 1.0)
    u = tf.random.uniform(
      shape=[batch, num_samples],dtype=dt)
    # for numerical stability
    max_logit = tf.math.reduce_max(logits, axis=1, keepdims=True)
    logits = logits - max_logit
    pdf = tf.cast(tf.math.exp(logits), dtype=dt)  # not normalized
    cdf = tf.math.cumsum(pdf, axis=1)
    cdf_last= cdf[:, -1:]
    u = u * cdf_last
    if num_samples == 0 or batch == 0:
        # A tf.searchsorted bug workaround
        return tf.zeros([batch, num_samples])
    else:
        return tf.searchsorted(cdf, u, side="right")
    
def normalize_graph(graph,
                    normalized = True,
                    add_self_loops = True):
    """Normalized the graph's adjacency matrix in the scipy sparse matrix format.

    Args:
    graph: A scipy sparse adjacency matrix of the input graph.
    normalized: If True, uses the normalized Laplacian formulation. Otherwise,
      use the unnormalized Laplacian construction.
    add_self_loops: If True, adds a one-diagonal corresponding to self-loops in
      the graph.

    Returns:
    A scipy sparse matrix containing the normalized version of the input graph.
    """
    if add_self_loops:
        graph = graph + scipy.sparse.identity(graph.shape[0])
        degree = np.squeeze(np.asarray(graph.sum(axis=1)))
    if normalized:
        with np.errstate(divide='ignore'):
              inverse_sqrt_degree = 1. / np.sqrt(degree)
        inverse_sqrt_degree[inverse_sqrt_degree == np.inf] = 0
        inverse_sqrt_degree = scipy.sparse.diags(inverse_sqrt_degree)
        return inverse_sqrt_degree @ graph @ inverse_sqrt_degree
    else:
        with np.errstate(divide='ignore'):
              inverse_degree = 1. / degree
        inverse_degree[inverse_degree == np.inf] = 0
        inverse_degree = scipy.sparse.diags(inverse_degree)
        return inverse_degree @ graph