import argparse
import ast

def parse_dict(value):
    try:
        return ast.literal_eval(value)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")
def load_config(parser):
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    parser.add_argument('--wait_time', type=float, default=0 * 60 * 60)
    parser.add_argument('--gpu_chose', type=int, default=0)
    parser.add_argument('--use_model_name', type=str, default='EDDM')

    # model
    parser.add_argument('--num_channels', type=int, default=64, help='number of initial channels in denosing model')
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--z_emb_dim', type=int, default=100)
    parser.add_argument('--z_emb_channels', nargs='+', type=int, default=[256, 256, 256, 256])
    parser.add_argument('--t_emb_dim', type=int, default=64)
    parser.add_argument('--t_emb_channels', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--level_channels', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4], help='channel multiplier')
    parser.add_argument('--attn_levels', default=(16,))
    parser.add_argument('--use_cross_attn', action='store_true', default=False)
    parser.add_argument('--num_resblocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='type of resnet block, choice in biggan and ddpm')
    parser.add_argument('--use_tanh_final', action='store_false', default=True)
    parser.add_argument('--phase', type=str, default='train_cell', help='model train_cell, train_snn or test_snn')
    parser.add_argument('--output_complete', action='store_true', default=True)
    parser.add_argument('--use_multi_flow', action='store_true', default=True)
    parser.add_argument('--driving_flow', type=float, default=0.0)
    parser.add_argument('--network_type', default='normal', help='choose of normal, large, max')

    # data
    parser.add_argument('--image_size', type=int, default=256, help='size of image')
    parser.add_argument('--input_channels', type=int, default=3, help='channel of image')
    parser.add_argument('--input_path', default='', help='path to input data')
    parser.add_argument('--checkpoint_path', default='', help='path to output saves')
    parser.add_argument('--normed', action='store_true', default=False)
    parser.add_argument('--source', type=str, default='T1', help='contrast selection for model')
    parser.add_argument('--target', type=str, default='T2', help='contrast selection for model')

    # training
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='learning rate')
    parser.add_argument('--lrf', type=float, default=1e-5, help='learning rate final')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=True, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=5, help='save content')
    parser.add_argument('--save_ckpt_every', type=int, default=5, help='save ckpt every x epochs')
    parser.add_argument('--lambda_l1', type=float, default=1, help='weightening of loss')
    parser.add_argument('--lambda_l2', type=float, default=1, help='weightening of loss')
    parser.add_argument('--lambda_perceptual', type=float, default=1, help='weightening of loss')
    parser.add_argument('--log_iteration', type=int, default=100)
    parser.add_argument('--use_reg', action='store_true', default=False)

    # val
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--val_every', type=int, default=5, help='validation every x epochs')
    parser.add_argument('--val_batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--which_epoch', type=int, default=120)
    parser.add_argument('--sample_fixed', action='store_true', default=False)

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=4, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')
    parser.add_argument('--port_num', type=str, default='6021', help='port selection for code')

    # cell
    parser.add_argument('--cell_num', type=int, default=4)
    parser.add_argument('--cell_capacity', type=float, default=1)
    parser.add_argument('--cell_time_mult', type=int, default=1)
    parser.add_argument('--sqrt_capacity', action='store_true', default=True)
    parser.add_argument('--train_xi', nargs='+', type=float, default=[0.0, 0.12, 0.43, 0.86, 1.00])
    parser.add_argument('--train_zeta', nargs='+', type=float, default=[0.0, 0.12, 0.43, 0.86, 1.00])
    parser.add_argument('--pulses_type', type=str, default='xt_pulses', help='noise_pulses or xt_pulses')

    # snn
    parser.add_argument('--snn_batch_size', type=int, default=8)
    parser.add_argument('--snn_dataset_num', type=int, default=30)
    parser.add_argument('--snn_sleep_time', type=int, default=5)
    parser.add_argument('--snn_log_time', type=int, default=500)
    parser.add_argument('--snn_env_step', type=int, default=20000)
    parser.add_argument('--prepare_data_num', type=int, default=20)
    # id : [time_layer, [input_ids], intensity, capacity]
    parser.add_argument('--snn_name', type=str, default="line")
    parser.add_argument('--snn_init_search', action='store_true', default=True)
    parser.add_argument('--snn_line', type=parse_dict, default={0.00: [-1, [0.10], 0.0, 1.0],
                                                                0.10: [0, [0.20], 0.22793515297599415, 0.992664635181427],
                                                                0.20: [1, [0.30], 0.28139371247105704, 0.90867680311203],
                                                                0.30: [2, [1.00], 0.28836720723156123, 0.5130777806043625],
                                                                1.00: [3, [-1], 1.0, 1.0],
                                                                })

    parser.add_argument('--snn_vp', type=parse_dict, default={0.00: [-1, [0.10], 0.0, 1.0],
                                                                0.10: [0, [0.20], 0.2016, 0.8674],
                                                                0.20: [1, [0.30], 0.1930, 0.4622],
                                                                0.30: [2, [1.00], 0.1119, 0.1697],
                                                                1.00: [3, [-1], 1.0, 1.0],
                                                                })

    parser.add_argument('--snn_double_tfw', type=parse_dict, default={
         0.0: [-1, [0.1], 0.0227, 1.0094],
         0.1: [0, [0.2], 0.4056, 0.9521],
         0.2: [1, [0.3, 0.31], 0.1554, 0.7791],
         0.3: [2, [1.0, 1.01, 1.02], 0.1554, 0.7791],
         0.31: [2, [1.0, 1.01, 1.02], 0.1554, 0.7661],
         1.0: [3, [-1], 1.0082, 0.9154],
         1.01: [3, [-1], 1.0082, 0.9154],
         1.02: [3, [-1], 1.0082, 0.9154]})

    parser.add_argument('--snn_11111_llvip', type=parse_dict, default={
        0.0: [-1, [0.1], 0, 1],
        0.1: [0, [0.2], 0.18554755803061238, 0.9971982687711716],
        0.2: [1, [0.3], 0.22149885224080634, 0.9424884617328644],
        0.3: [2, [1.0], 0.19593817478853817, 0.5349813103675842],
        1.0: [3, [-1], 1, 1]})

    parser.add_argument('--snn_11112_llvip', type=parse_dict, default={
        0.00: [-1, [0.10], 0, 1],
        0.10: [0, [0.20], 0.18554755803061238, 0.9971982687711716],
        0.20: [1, [0.30], 0.22149885224080634, 0.9424884617328644],
        0.30: [2, [1.00, 1.01], 0.19593817478853817, 0.5349813103675842],
        1.00: [3, [-1], 1, 1],
        1.01: [3, [-1], 1, 1]})

    parser.add_argument('--snn_11123_llvip', type=parse_dict, default={
        0.00: [-1, [0.10], 0, 1],
        0.10: [0, [0.20], 0.18554755803061238, 0.9971982687711716],
        0.20: [1, [0.30, 0.31], 0.22149885224080634, 0.9424884617328644],
        0.30: [2, [1.00, 1.01, 1.02], 0.19593817478853817, 0.5349813103675842],
        0.31: [2, [1.00, 1.01, 1.02], 0.19593817478853817, 0.5349813103675842],
        1.00: [3, [-1], 1, 1],
        1.01: [3, [-1], 1, 1],
        1.02: [3, [-1], 1, 1],})

    parser.add_argument('--snn_11234_llvip', type=parse_dict, default={
        0.00: [-1, [0.10], 0, 1],
        0.10: [0, [0.20], 0.18554755803061238, 0.9971982687711716],
        0.20: [1, [0.30, 0.31], 0.22149885224080634, 0.9424884617328644],
        0.21: [1, [0.30, 0.31], 0.22149885224080634, 0.9424884617328644],
        0.30: [2, [1.00, 1.01, 1.02, 1.03], 0.19593817478853817, 0.5349813103675842],
        0.31: [2, [1.00, 1.01, 1.02, 1.03], 0.19593817478853817, 0.5349813103675842],
        0.32: [2, [1.00, 1.01, 1.02, 1.03], 0.19593817478853817, 0.5349813103675842],
        1.00: [3, [-1], 1, 1],
        1.01: [3, [-1], 1, 1],
        1.02: [3, [-1], 1, 1],
        1.03: [3, [-1], 1, 1],})

    parser.add_argument('--snn_double_brats1to2', type=parse_dict, default={
        0.0: [-1, [0.1], 0.0009, 1.0001],
        0.1: [0, [0.2], 0.2225, 0.9538],
        0.2: [1, [0.3, 0.31], 0.1309, 0.86],
        0.3: [2, [1.0, 1.01, 1.02], 1.0017, 0.5559],
        0.31: [2, [1.0, 1.01, 1.02], 1.0017, 0.5559],
        1.0: [3, [-1], 1.0885, 1.9943],
        1.01: [3, [-1], 1.0885, 1.9943],
        1.02: [3, [-1], 1.0885, 1.9943]})

    parser.add_argument('--snn_double_brats2to1', type=parse_dict, default={
        0.0: [-1, [0.1], 0.0035, 1.0001],
        0.1: [0, [0.2], 0.2665, 1.1103],
        0.2: [1, [0.3, 0.31], 0.2834, 0.6786],
        0.3: [2, [1.0, 1.01, 1.02], 0.2834, 0.6786],
        0.31: [2, [1.0, 1.01, 1.02], 0.3808, 0.4197],
        1.0: [3, [-1], 1.0037, 0.557],
        1.01: [3, [-1], 1.0037, 0.557],
        1.02: [3, [-1], 1.0037, 0.557]})

    parser.add_argument('--snn_double_oasis1to2', type=parse_dict, default={
        0.0: [-1, [0.1], -0.0083, 0.9985],
        0.1: [0, [0.2], 0.2411, 0.9961],
        0.2: [1, [0.3, 0.31], 0.2173, 0.8005],
        0.3: [2, [1.0, 1.01, 1.02], 0.2173, 0.8005],
        0.31: [2, [1.0, 1.01, 1.02], 0.1489, 0.5322],
        1.0: [3, [-1], 0.9976, 0.4635],
        1.01: [3, [-1], 0.9976, 0.4635],
        1.02: [3, [-1], 0.9976, 0.4635]})

    parser.add_argument('--snn_double_oasis2to1', type=parse_dict, default={
        0.0: [-1, [0.1], 0.0023, 0.9998],
        0.1: [0, [0.2], 0.0032, 1.2512],
        0.2: [1, [0.3, 0.31], 0.2718, 0.3486],
        0.3: [2, [1.0, 1.01, 1.02], 0.2718, 0.3486],
        0.31: [2, [1.0, 1.01, 1.02], 1.9368, 0.045],
        1.0: [3, [-1], 1.0144, 1.6224],
        1.01: [3, [-1], 1.0144, 1.6224],
        1.02: [3, [-1], 1.0144, 1.6224]})


    parser.add_argument('--snn_double', type=parse_dict, default={  0.00: [-1, [0.10], 0.0, 1.0],
                                                                    0.10: [0, [0.20], 0.2263, 0.9927],
                                                                    0.20: [1, [0.30, 0.31], 0.2557, 0.9087],
                                                                    0.30: [2, [1.00, 1.01, 1.02], 0.1480, 0.5131],
                                                                    0.31: [2, [1.00, 1.01, 1.02], 0.1480, 0.5131],
                                                                    1.00: [3, [-1], 1.0, 1.0],
                                                                    1.01: [3, [-1], 1.0, 1.0],
                                                                    1.02: [3, [-1], 1.0, 1.0],
                                                            })
    parser.add_argument('--snn_triple', type=parse_dict, default={0.0: [0, [1.0, 1.1, 1.2], 0., 1.0],
                                                                  1.0: [1, [2.0, 2.1, 2.2], 0.23, 1.0],
                                                                  1.1: [1, [2.0, 2.1, 2.2], 0.23, 1.0],
                                                                  1.2: [1, [2.0, 2.1, 2.2], 0.23, 1.0],
                                                                  2.0: [2, [3.0, 3.1, 3.2], 0.33, 0.9],
                                                                  2.1: [2, [3.0, 3.1, 3.2], 0.33, 0.9],
                                                                  2.2: [2, [3.0, 3.1, 3.2], 0.33, 0.9],
                                                                  3.0: [3, [-1], 0.33, 0.5],
                                                                  3.1: [3, [-1], 0.33, 0.5],
                                                                  3.2: [3, [-1], 0.33, 0.5],
                                                                  })

    parser.add_argument('--snn_line9', type=parse_dict, default={0.00: [-1, [0.10], 0.0, 1.0],
                                                                 0.10: [0, [0.20], 0.3661, 0.9996],
                                                                 0.20: [1, [0.30], 0.4772, 0.9979],
                                                                 0.30: [1, [0.40], 0.5252, 0.9928],
                                                                 0.40: [1, [0.50], 0.5458, 0.9772],
                                                                 0.50: [1, [0.60], 0.5445, 0.9353],
                                                                 0.60: [1, [0.70], 0.5119, 0.8381],
                                                                 0.70: [1, [0.80], 0.4374, 0.6650],
                                                                 0.80: [2, [1.00], 0.3350, 0.4536],
                                                                 1.00: [3, [-1], 1.0, 1.0],
                                                                })
    return parser
