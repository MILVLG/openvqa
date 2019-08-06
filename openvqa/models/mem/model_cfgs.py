# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        # memory parameters
        self.LAYER = 6
        self.HIDDEN_SIZE = 512
        self.BBOXFEAT_EMB_SIZE = 2048
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024

        # memory parameters
        self.VALUE_LR_TIMES = 4
        self.MEM_SHARE_VALUES = False
        self.MEM_SPARSE = False
        self.SUB_SIZE = 512
        self.MEM_SIZE = self.SUB_SIZE * self.SUB_SIZE
        # self.modulo_size = False
        # self.n_indices = params.n_indices
        self.K_DIM = 512 // 2
        # self.v_dim = params.mem_v_dim if params.mem_v_dim > 0 else output_dim
        self.MEM_HEAD = 4
        self.KNN = 32
        # self.shuffle_indices = params.mem_shuffle_indices
        self.KEYS_NORMALIZED_INIT = False
        # self.product_quantization = params.mem_product_quantization
        # assert self.modulo_size == - \
        #     1 and self.size == self.n_indices or self.n_indices > self.size == self.modulo_size >= 1

        self.KEYS_TYPE = "uniform"
        # self.learn_keys = True
        self.USE_DIFFERENT_KEYS = True
        self.QUERY_DETACH_INPUT = False
        # self.query_net_learn = True
        # self.multi_query_net = False
        # self.shuffle_query = params.mem_shuffle_query

        # assert self.use_different_keys is False or self.keys_type in [
        #     'gaussian', 'uniform']
        # assert self.use_different_keys is False or self.heads >= 2 or self.product_quantization
        # assert self.multi_query_net is False or self.heads >= 2 or self.product_quantization
        # assert self.shuffle_query is False or self.heads > 1 and params.mem_query_layer_sizes == ''
        # assert self.shuffle_query is False or self.input_dim % (
        #     2 ** self.heads) == 0

        # self.normalize_query = False
        # self.temperature = params.mem_temperature
        self.SCORE_SOFTMAX = True
        # self.score_subtract = params.mem_score_subtract
        # self.score_normalize = params.mem_score_normalize
        # assert self.score_subtract in ['', 'min', 'mean', 'median']
        # assert self.score_subtract == '' or self.knn >= 2
        # assert not (
        #     self.score_normalize and self.score_softmax and self.score_subtract == '')

        # dropout
        self.INPUT_DROPOUT = 0.1
        self.QUERY_DROPOUT = 0
        self.VALUE_DROPOUT = 0

        # parser.add_argument("--mem_implementation", type=str, default="pq_fast",
        #                     help="Memory implementation (flat, pq_default, pq_fast)")

        # # optimization
        # parser.add_argument("--mem_grouped_conv", type=bool_flag, default=False,
        #                     help="Use grouped convolutions in the query network")
        # parser.add_argument("--mem_values_optimizer", type=str, default="adam,lr=0.001",
        #                     help="Memory values optimizer ("" for the same optimizer as the rest of the model)")
        # parser.add_argument("--mem_sparse", type=bool_flag, default=False,
        #                     help="Perform sparse updates for the values")

        # # global parameters
        # parser.add_argument("--mem_input2d", type=bool_flag, default=False,
        #                     help="Convolutional query network")
        # parser.add_argument("--mem_k_dim", type=int, default=256,
        #                     help="Memory keys dimension")
        # parser.add_argument("--mem_v_dim", type=int, default=-1,
        #                     help="Memory values dimension (-1 for automatic output dimension)")
        # parser.add_argument("--mem_heads", type=int, default=4,
        #                     help="Number of memory reading heads")
        # parser.add_argument("--mem_knn", type=int, default=32,
        #                     help="Number of memory slots to read / update - k-NN to the query")
        # parser.add_argument("--mem_share_values", type=bool_flag, default=False,
        #                     help="Share values across memories")
        # parser.add_argument("--mem_shuffle_indices", type=bool_flag, default=False,
        #                     help="Shuffle indices for different heads")
        # parser.add_argument("--mem_shuffle_query", type=bool_flag, default=False,
        #                     help="Shuffle query dimensions (when the query network is the identity and there are multiple heads)")
        # parser.add_argument("--mem_modulo_size", type=int, default=-1,
        #                     help="Effective memory size: indices are taken modulo this parameter. -1 to disable.")

        # # keys
        # parser.add_argument("--mem_keys_type", type=str, default="uniform",
        #                     help="Memory keys type (binary,gaussian,uniform)")
        # parser.add_argument("--mem_n_keys", type=int, default=512,
        #                     help="Number of keys")
        # parser.add_argument("--mem_keys_normalized_init", type=bool_flag, default=False,
        #                     help="Normalize keys at initialization")
        # parser.add_argument("--mem_keys_learn", type=bool_flag, default=True,
        #                     help="Learn keys")
        # parser.add_argument("--mem_use_different_keys", type=bool_flag, default=True,
        #                     help="Use different keys for each head / product quantization")

        # # queries
        # parser.add_argument("--mem_query_detach_input", type=bool_flag, default=False,
        #                     help="Detach input")
        # parser.add_argument("--mem_query_layer_sizes", type=str, default="0,0",
        #                     help="Query MLP layer sizes ('', '0,0', '0,512,0')")
        # parser.add_argument("--mem_query_kernel_sizes", type=str, default="",
        #                     help="Query MLP kernel sizes (2D inputs only)")
        # parser.add_argument("--mem_query_bias", type=bool_flag, default=True,
        #                     help="Query MLP bias")
        # parser.add_argument("--mem_query_batchnorm", type=bool_flag, default=False,
        #                     help="Query MLP batch norm")
        # parser.add_argument("--mem_query_net_learn", type=bool_flag, default=True,
        #                     help="Query MLP learn")
        # parser.add_argument("--mem_query_residual", type=bool_flag, default=False,
        #                     help="Use a bottleneck with a residual layer in the query MLP")
        # parser.add_argument("--mem_multi_query_net", type=bool_flag, default=False,
        #                     help="Use multiple query MLP (one for each head)")

        # # values initialization
        # parser.add_argument("--mem_value_zero_init", type=bool_flag, default=False,
        #                     help="Initialize values with zeros")

        # # scoring
        # parser.add_argument("--mem_normalize_query", type=bool_flag, default=False,
        #                     help="Normalize queries")
        # parser.add_argument("--mem_temperature", type=float, default=1,
        #                     help="Divide scores by a temperature")
        # parser.add_argument("--mem_score_softmax", type=bool_flag, default=True,
        #                     help="Apply softmax on scores")
        # parser.add_argument("--mem_score_subtract", type=str, default="",
        #                     help="Subtract scores ('', min, mean, median)")
        # parser.add_argument("--mem_score_normalize", type=bool_flag, default=False,
        #                     help="L1 normalization of the scores")

        # # dropout
        # parser.add_argument("--mem_input_dropout", type=float, default=0,
        #                     help="Input dropout")
        # parser.add_argument("--mem_query_dropout", type=float, default=0,
        #                     help="Query dropout")
        # parser.add_argument("--mem_value_dropout", type=float, default=0,
        #                     help="Value dropout")
