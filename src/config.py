from shared_configs import ModelConfig, DataConfig

e = 2.71828


class REHGLConfig(ModelConfig):
    def __init__(self, dataset, seed=0):
        super(REHGLConfig, self).__init__('REHGL')
        default_settings = \
            {'acm': {'alpha': 0.2, 'dropout': 0.5, 'fgh_th': 0.99, 'fgp_th': 0.2, 'sem_th': 0.6,
                     'mp_list': ['psp', 'pap', 'pspap']},
             'yelp': {'alpha': 0.2, 'dropout': 0.2, 'fgh_th': 0.4, 'fgp_th': 0.99, 'sem_th': 0.2,
                      'mp_list': ['bub', 'bsb', 'bublb', 'bubsb']},
             'imdb': {'alpha': 1, 'dropout': 0, 'fgh_th': 0.2, 'fgp_th': 0.2, 'sem_th': 0.6,
                      'mp_list': ['mdm', 'mam']}
             }
        self.dataset = dataset
        self.__dict__.update(default_settings[dataset])
        # ! Model settings
        self.lr = 0.01
        self.seed = seed
        self.save_model_conf_list()  # * Save the model config list keys
        self.conv_method = 'gcn'
        self.num_head = 2
        self.early_stop = 80
        self.adj_norm_order = 1
        self.feat_norm = -1
        self.emb_dim = 64   # GCN-hid
        self.com_feat_dim = 64   # graph-gen-projection
        self.weight_decay = 5e-4
        self.model = 'REHGL'
        self.epochs = 200
        self.exp_name = 'debug'
        self.save_weights = True
        d_conf = DataConfig(dataset)
        self.__dict__.update(d_conf.__dict__)
