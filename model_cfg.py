# Model setup cfg

def init_cfg():
    '''
    Dnet baseline setup
    '''
    model_cfg = {
        'general': {
            'overfit': False,
            'sparse_pretrain_epochs': 50,
            'dense_pretrain_epochs': 40,
        },
        'sparse_backbone': {
            'num_features': 30,
        },
        'rpn': {
            'num_features': 30,
            'min_conf_train': 0.3,
            'max_proposals_train': 6,
            'min_conf_test': 0.6,
            'max_proposals_test': 6,
        },
        'dense_backbone': {
            'num_features': 30,
            'fixed_size': [192, 96, 192],
            'pad_factor': None,
        },
        'completion': {
            'num_features': 30,
            'bbox_shape': [8, 8, 8],
            'gt_augm': True,
            'total_weight': 2,
        },
        'nocs': {
            'num_features': 30,
            'bbox_shape': [8, 8, 8],
            'gt_augm': True,
            'weights': {
                'noc': 1.0,
                'rot': 0.05,
                'transl': 0.01,
                'scale': 0.01,
            },
            'noc_samples': 300,
        },
    }
    return model_cfg
