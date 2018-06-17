import argparse, pprint, random, datetime, dateutil.tz, time, numpy as np
import torch, torchvision.transforms as transforms
from config import cfg_from_file, cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate a recurrent GAN, which generated images'
                                                 ' from captions.')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/train_birds.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='1,3,4,5,6,7')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)

    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = './output/%s_%s' % (cfg.DATASET_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
            bshuffle = False
            split_dir = 'test'

    # Get data loader
    num_time_step = np.log2(cfg.FINAL_IMAGE_SIZE / cfg.INITIAL_IMAGE_SIZE)

    imsize = cfg.INITIAL_IMAGE_SIZE * (2 ** int(num_time_step))

    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    if cfg.DATASET_NAME == 'birds':
        from datasets import BirdsDataset

        dataset = BirdsDataset(cfg.DATA_DIR, split_dir,
                              base_size=cfg.INITIAL_IMAGE_SIZE,
                              transform=image_transform)
    # elif cfg.DATASET_NAME == 'flowers':
    #     from datasets import FlowersDataset
    #
    #     dataset = FlowersDataset(cfg.DATA_DIR, split_dir,
    #                              base_size=cfg.INITIAL_IMAGE_SIZE,
    #                              transform=image_transform)
    assert dataset

    num_gpu = len(cfg.GPU_ID.split(','))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    from trainer import RecurrentGANTrainer as trainer
    algo = trainer(output_dir, dataloader, imsize)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate(split_dir)
    end_t = time.time()
    print 'Total time for training:', end_t - start_t
