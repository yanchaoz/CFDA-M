from torch.utils.data import DataLoader
from torch import optim
from dataset import Dataset
from models.networks import define_S
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--pre_train_path', type=str, default='/***/***')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--save_freq', type=int, default=500)
parser.add_argument('--root', type=str, default='/***/***')


def train():
    model = define_S(1, 1, 64, 'duseunet', "instance", not True, "xavier", 0.02, False, False, [0])
    para = torch.load(args.pre_train_path, map_location=lambda storage, loc: storage.cuda())
    model.load_state_dict(para)
    model.cuda()
    model.train()
    model_ema = define_S(1, 1, 64, 'duseunet', "instance", not True, "xavier", 0.02, False, False, [0])
    model_ema.train()
    model.cuda()

    for param in model_ema.parameters():
        param.detach_()

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dataset_train = Dataset(args.root, co_transform=co_trnsform, transform=x_transforms)
    dataloaders_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    for total_counter in range(6001):
        try:
            x, y = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(dataloaders_train)
            x, y = unlabeled_train_iter.next()

        model_ema = update_ema_variables(model, model_ema, alpha=0.999, global_step=total_counter)

        input_t = x.cuda()
        input_s = y.cuda()
        optimizer.zero_grad()

        output_t = model_ema(input_t)

        input_t_trans, rot_mask, flip_mask = transforms_for_rot(input_t)
        output_t_trans = model_ema(input_t_trans)
        output_t_inv = transforms_back_rot(output_t_trans, rot_mask, flip_mask)

        output_fusion = (output_t_inv + output_t) / 2

        output_fusion[output_fusion >= 0.5] = 1
        output_fusion[output_fusion < 0.5] = 0

        input_s_c, output_fusion_c, _ = Cutout(input_s, output_fusion, n_holes=1, length=32)
        output_s = model(input_s_c)

        consistency_weight = get_current_consistency_weight(total_counter // 150)

        loss = criterion(output_s, output_fusion_c) * consistency_weight
        loss.backward()
        optimizer.step()

        print(loss.item(), total_counter)

        if ((total_counter + 1) % args.save_freq == 0) & (total_counter != 0):
            torch.save(model.state_dict(), './checkpoints/weights_%d.pth' % (total_counter // args.save_freq),
                       _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    args = parser.parse_args()
    train()
