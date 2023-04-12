import time
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils.utils import *
from model import Net
from einops import rearrange
import math

# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upfactor", type=int, default=4, help="upscale factor")
    parser.add_argument('--model_name', type=str, default='DistgSSR_4xSR')
    parser.add_argument('--trainset_dir', type=str, default='./Data/Train_4xSR_5x5/')
    parser.add_argument('--testset_dir', type=str, default='./Data/Test_4xSR_5x5/')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument('--crop', type=bool, default=True, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--patchsize", type=int, default=64, help="")
    parser.add_argument("--minibatch", type=int, default=4, help="LFs are cropped into patches to save GPU memory")

    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./log/DistgSSR_4xSR_epoch_1.pth.tar')

    return parser.parse_args()


  
def train(cfg, train_loader, test_Names, test_loaders):

    net = Net(cfg.angRes, cfg.upfactor)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            epoch_state = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.model_path))

    if cfg.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1])
    
    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler._step_count = epoch_state
    loss_epoch = []
    loss_list = []

    for idx_epoch in range(epoch_state, cfg.n_epochs):
        for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            out = net(data.to(cfg.device))
            loss = criterion_Loss(out, label.to(cfg.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            if cfg.parallel:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                }, save_path='./log/', filename=cfg.model_name + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            else:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                }, save_path='./log/', filename=cfg.model_name + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []

        ''' evaluation '''
        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []
            for index, test_name in enumerate(test_Names):
                test_loader = test_loaders[index]
                psnr_epoch_test, ssim_epoch_test = valid(test_loader, net)
                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
                pass
            pass

        scheduler.step()
        pass


def valid(test_loader, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze().to(cfg.device)

        if cfg.crop == False:
            with torch.no_grad():
                outLF = net(data.unsqueeze(0).unsqueeze(0).to(cfg.device))
                outLF = outLF.squeeze()
        else:
            lf_lr = rearrange(data.squeeze(), '(u h) (v w) -> u v h w', u=cfg.angRes, v=cfg.angRes)
            patchsize = cfg.patchsize
            stride = patchsize // 2
            sub_lfs = LFdivide(lf_lr, patchsize, stride)

            n1, n2, u, v, c, h, w = sub_lfs.shape
            sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) c (u h) (v w)')
            mini_batch = cfg.minibatch
            num_inference = (n1 * n2) // mini_batch
            with torch.no_grad():
                out_lfs = []
                for idx_inference in range(num_inference):
                    input_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :]
                    out_lfs.append(net(input_lfs.to(cfg.device)))
                if (n1 * n2) % mini_batch:
                    input_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :]
                    out_lfs.append(net(input_lfs.to(cfg.device)))

            out_lfs = torch.cat(out_lfs, dim=0)
            out_lfs = rearrange(out_lfs, '(n1 n2) c (u h) (v w) -> n1 n2 u v c h w', n1=n1, n2=n2, u=cfg.angRes,
                                v=cfg.angRes)
            outLF = LFintegrate(out_lfs, patchsize * cfg.upfactor, patchsize * cfg.upfactor // 2)
            outLF = outLF[:, :, 0: lf_lr.shape[2] * cfg.upfactor, 0: lf_lr.shape[3] * cfg.upfactor]

        psnr, ssim = cal_metrics(label.to(cfg.device), outLF, cfg.angRes)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


if __name__ == '__main__':
    cfg = parse_args()
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    train(cfg, train_loader, test_Names, test_Loaders)
