def arguments_parsing():
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('-s', '--seed', type=int, default=1126)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-e', '--num_epoch', type=int, default=14)
    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-3)

    # model
    parser.add_argument('-dev', '--device', type=int, default=0)

    # I/O
    parser.add_argument('-dp', '--data_path', type=str, default='./data/train.csv')
    parser.add_argument('-mp', '--model_path', type=str, default='./model')
    parser.add_argument('-si', '--save_interval', type=int, default=0)
    parser.add_argument('-o', '--output_file', type=str, default='./output.csv')

    # Mode
    parser.add_argument('-m', '--mode', type=str, choices=['training', 'validation', 'testing'], default='training')

    return parser.parse_args()

if __name__ == '__main__':
    args = arguments_parsing()
    args.device = torch.device('cpu') if args.device < 0 else torch.device('cuda:%d' % (args.device))

    pkeys = ['seed']
    args.config = str({k:vars(args)[k] for k in pkeys})

    # Reproduction Settings
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
