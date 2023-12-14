import argparse
import torch
import torch.nn as nn
from model import *
from load_data import *
from utils import *
import torch.nn.functional as F
import scipy.io as sio

seed_setting(seed=2026)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# parameter setting
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Office-Home', choices=['Office-Home', 'Office-31', 'Digits'])
    parser.add_argument('--nbit', type=int, default=64, choices=[16, 32, 64, 128])
    parser.add_argument('--batchsize', type=int, default=256) # 20230725 128 -》256
    parser.add_argument('--num_epoch', type=int, default=70) # 45
    parser.add_argument('--inter', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--w', type=float, default=8e-6)
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")
    parser.add_argument('--lamda1', type=float, default=1)
    parser.add_argument('--lamda2', type=float, default=1)
    parser.add_argument('--lamda3', type=float, default=100)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--ratio2', type=float, default=0.5)
    parser.add_argument('--ratio3', type=float, default=0.9) # 1
    parser.add_argument('--domain', type=str, default='ArtToReal_World') # ArtToReal_World AmazonToDslr

    # print
    args = parser.parse_args()
    print(args)

    return args

def train(args, source_loader, target_train_loader, target_test_loader, n_class, dim1):
    # 1. define modal
    pchNet = PCHModel(args, n_class, dim1)
    pchNet.cuda()

    # 2. define loss
    criterion_l2 = nn.MSELoss().cuda()
    criterion = torch.nn.CrossEntropyLoss()

    # 3. define optimizer
    optimizer = torch.optim.Adam(pchNet.parameters(), args.lr, weight_decay=args.w)

    # 4. set model to train model
    pchNet.train()

    # 5. set loader
    len_source_loader = len(source_loader)  # 获取源域数据加载器中batch的数量 19
    len_target_loader = len(target_train_loader)  # 31
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    # 6. define variables
    memory_source_centers = torch.zeros(n_class, dim1).cuda()
    memory_target_centers = torch.zeros(n_class, dim1).cuda()

    # 7. train
    for epoch in range(1, args.num_epoch + 1):

        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        correct = 0
        total_tgt = 0

        for step in range(n_batch):
            optimizer.zero_grad()

            # 得到每个batch sample
            data_source, label_source, _ = next(iter_source)  # .next()
            data_target, label_target, _ = next(iter_target)  # .next()

            data_source = data_source.cuda()
            data_target = data_target.cuda()

            label_source = label_source.cuda()
            label_target = label_target.cuda() # for eval

            # forward
            source_feat, source_h, target_feat, target_h = pchNet(data_source, data_target)

            # 计算源域的原型
            batch_source_centers = compute_centers(source_feat, label_source.squeeze(), n_class)

            memory_source_centers = memory_source_centers.detach() + batch_source_centers

            # 对聚类中心进行 L2 归一化，以确保每个聚类中心向量的长度为1
            memory_source_centers = F.normalize(memory_source_centers, dim=1)

            # 计算目标域的伪标签和原型
            target_plabel = psedo_labeling(n_class, target_feat, memory_source_centers, label_target)
            target_plabel = target_plabel.cuda()

            batch_target_centers = compute_centers(target_feat, target_plabel, n_class)

            memory_target_centers = memory_target_centers.detach() + batch_target_centers
            memory_target_centers = F.normalize(memory_target_centers, dim=1)

            ############# 0517 test  start #####################
            # batch
            correct_sample = torch.sum(target_plabel.unsqueeze(1) == label_target)
            n_sample = label_target.size(0)


            # for total
            correct += correct_sample
            total_tgt = total_tgt + n_sample

            # (1) prototype loss
            # TODO
            cluster_batch_loss = args.lamda1 * compute_cluster_loss(memory_source_centers, memory_target_centers, args.temperature, target_plabel, n_class)

            # (2) 量化loss
            source_b = torch.sign(source_h)
            target_b = torch.sign(target_h)
            sign_loss = args.ratio2 * args.lamda2 * criterion_l2(source_h, source_b) + (1 - args.ratio2) * args.lamda2 * criterion_l2(target_h, target_b)

            # (3) 关系保留loss
            # 源域关系
            label_source_onehot = torch.eye(n_class)[label_source.squeeze(1), :]
            S_I = label_source_onehot.mm(label_source_onehot.t())
            S_I = S_I.cuda()  # [0, 1]

            h_norm_s = F.normalize(source_h)
            S_h_s = h_norm_s.mm(h_norm_s.t()) # [-1, 1]
            S_h_s[S_h_s < 0] = 0  # [0, 1]

            relation_recons_loss1 = criterion_l2(S_h_s, 1.1 * S_I) * args.lamda3

            # 目标域关系
            F_S = F.normalize(source_feat)
            F_T = F.normalize(target_feat)
            S_T_feat = F_S.mm(F_T.t()) # [0, 1]

            h_norm_s = F.normalize(source_h)
            h_norm_t = F.normalize(target_h)
            S_T_h = h_norm_s.mm(h_norm_t.t()) # [-1, 1]
            S_T_h[S_T_h < 0] = 0 # [0, 1]

            relation_recons_loss2 = criterion_l2(S_T_feat, S_T_h) * args.lamda3

            relation_recons_loss = args.ratio3 * relation_recons_loss1 + (1 - args.ratio3) * relation_recons_loss2

            # total loss
            loss = cluster_batch_loss + sign_loss + relation_recons_loss
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 and (step + 1) == n_batch:
                print('Epoch [%3d/%3d]: Total Loss: %.4f, loss1: %.4f, loss2: %.4f, loss3: %.4f' % (
                    epoch + 1, args.num_epoch,
                    loss.item(),
                    cluster_batch_loss.item(),
                    sign_loss.item(),
                    relation_recons_loss.item()
                ))


    # len_target_dataset = len(target_test_loader.dataset)
    acc = 100. * correct / total_tgt
    print('total acc: %.8f' %(acc))
       ############# 0517 test  end #####################

    # test
    performance_eval(pchNet, source_loader, target_test_loader)

def performance_eval(model, database_loader, query_loader):

    model.eval().cuda()
    re_BI, re_L, qu_BI, qu_L = compress(database_loader, query_loader, model)

    ## Save
    _dict = {
        'retrieval_B': re_BI,
        'L_db':re_L,
        'val_B': qu_BI,
        'L_te':qu_L,
    }
    sava_path = 'hashcode/HASH_' + args.dataset + '_' + str(args.nbit) + 'bits.mat'
    sio.savemat(sava_path, _dict)

    return 0

def compress(database_loader, query_loader, model):

    # retrieval
    re_BI = list([])
    re_L = list([])
    for _, (data_I, data_L, _) in enumerate(database_loader):
        with torch.no_grad():
            var_data_I = data_I.cuda()
            code_I = model.predict(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())

    # query
    qu_BI = list([])
    qu_L = list([])
    for _, (data_I, data_L, _) in enumerate(query_loader):
        with torch.no_grad():
            var_data_I = data_I.cuda()
            code_I = model.predict(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_L = np.array(re_L)

    qu_BI = np.array(qu_BI)
    qu_L = np.array(qu_L)

    return re_BI, re_L, qu_BI, qu_L

if __name__ == '__main__':
    args = get_args()

    # train
    print("begin train")
    source_domain = args.domain.split('To')[0]
    target_domain = args.domain.split('To')[1]

    print('source domain: ' + source_domain)
    print('target domain: ' + target_domain)

    # load source & target data
    if args.dataset == 'Office-Home':  # (1) Art (2) Clipart (3) Product (4) Real_World
        base_path = '/data/CuiHui/OfficeHome_mat/'
        source_loader, n_class, dim1 = get_loader_source(args.batchsize, base_path, source_domain)
        target_loader = get_loader_target(args.batchsize, base_path, target_domain)
        target_train_loader = target_loader['train']
        target_test_loader = target_loader['query']

    elif args.dataset == 'Office-31':  # (1) Amazon (2) Dslr (3) Webcam
        base_path = '/data/CuiHui/Office31_mat/'
        source_loader, n_class, dim1 = get_loader_source(args.batchsize, base_path, source_domain)
        target_loader = get_loader_target(args.batchsize, base_path, target_domain)
        target_train_loader = target_loader['train']
        target_test_loader = target_loader['query']

    elif args.dataset == 'Digits':
        base_path = '/data/CuiHui/Digits/'
        source_loader, n_class, dim1 = get_loader_source(args.batchsize, base_path, source_domain)
        target_loader = get_loader_target(args.batchsize, base_path, target_domain)
        target_train_loader = target_loader['train']
        target_test_loader = target_loader['query']
    else:
        raise Exception('No this dataset!')

    train(args, source_loader, target_loader['train'], target_loader['query'], n_class, dim1)

    print('lamda1: %.8f, lamda2: %.8f, lamda3: %.8f' % (args.lamda1, args.lamda2, args.lamda3))
    print("******************************************")
