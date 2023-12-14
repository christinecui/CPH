import torch
import torch.nn.functional as F
import faiss
from sklearn import metrics
import warnings
import numpy as np
from munkres import Munkres
import random
import os

def seed_setting(seed=2026):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# todo ok
def compute_centers(x, psedo_labels, num_cluster):
    n_samples = x.size(0)
    if len(psedo_labels.size()) > 1:
        # 如果伪标签 psedo_labels 是一个二维张量，则将其转置得到一个形状为 (n_samples, num_cluster) 的张量
        weight = psedo_labels.T
    else:
        # 如果伪标签 psedo_labels 是一个一维张量，则创建一个形状为 (num_cluster, n_samples) 的零张量 weight，
        # 并将其中第 psedo_labels[i] 行、第 i 列的元素设置为 1，表示第 i 个样本属于第 psedo_labels[i] 个聚类
        weight = torch.zeros(num_cluster, n_samples).to(x)  # L, N
        weight[psedo_labels, torch.arange(n_samples)] = 1

    weight = weight.float()
    # 对 weight 进行 L1 归一化，即将每一行的元素值都除以该行的元素之和，以确保每个聚类的权重之和为 1
    weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
    # 通过矩阵乘法 torch.mm(weight, x) 将 weight 和 x 相乘，得到每个聚类的样本特征的加权平均值，即聚类中心
    centers = torch.mm(weight, x)
    # 对聚类中心进行 L2 归一化，以确保每个聚类中心向量的长度为1
    centers = F.normalize(centers, dim=1)

    return centers


@torch.no_grad()
def psedo_labeling(num_cluster, batch_features, centers, batch_target):
    l2_normalize =True
    torch.cuda.empty_cache()

    # (1) 进行L2归一化
    if l2_normalize:
        batch_features = F.normalize(batch_features, dim=1)
        batch_features_cpu = batch_features.cpu()

    # (2) 计算batch_feat和source_centers的相似度矩阵
    # 为了使用F.cosine_similarity函数,需要将cluster_centers和centers分别扩展成大小为(c, 1, d)和(1, n, d)的三维张量,以便在第二维上进行广播操作
    # todo ok
    centers_cpu = centers.cpu()
    btarget_cen_similarity = F.cosine_similarity(batch_features_cpu.unsqueeze(1), centers_cpu.unsqueeze(0), dim=2)

    # (3) 得到源域和目标域标签的对应关系
    relation = torch.zeros(num_cluster, dtype=torch.int64) - 1
    # 对similarity中的每一行进行排序,并按照从大到小的顺序记录索引
    sorted_indices = torch.argsort(btarget_cen_similarity, dim=1, descending=True)
    new_cluster_labels = sorted_indices[:, 0]

    return new_cluster_labels


def clustering(features: torch.Tensor, n_clusters: int):
    """
    对输入的features进行聚类，返回聚类后的标签和聚类中心。
    Args:
        features: 输入的特征数据，类型为torch.Tensor，大小为(batch_size, feature_dim)，batch_size表示数据的数量，
                  feature_dim表示每个数据的特征维度。
        n_clusters: 聚类的数量。
    Returns:
        聚类后的标签和聚类中心，类型为元组(torch.Tensor, torch.Tensor)。
        plabels：聚类后的标签，类型为torch.Tensor，大小为(batch_size,)，batch_size表示数据的数量。
        centroids：聚类中心，类型为torch.Tensor，大小为(n_clusters, feature_dim)，n_clusters表示聚类的数量，
                   feature_dim表示每个数据的特征维度。
    """
    # 将数据转换为numpy数组，并转换为float32类型
    x_np = features.numpy().astype('float32')

    # 初始化Faiss的KMeans聚类器
    dim = features.shape[1]
    kmeans = faiss.Kmeans(d=dim, k=n_clusters, seed=2023, gpu=1, niter=100, verbose=False, nredo=5,
                          min_points_per_centroid=1, spherical=True)

    # 使用KMeans算法对数据进行聚类
    # kmeans.train(x_np)
    kmeans.train(x_np, init_centroids=None)

    # 将聚类中心转换为PyTorch的tensor类型
    centroids_np = kmeans.centroids
    centroids = torch.from_numpy(centroids_np)

    # 将数据分配到最近的聚类中心 I: 距离最近的质心 D：L2距离 D, I = kmeans.index.search(x, 1)
    _, plabels_np = kmeans.index.search(x_np, 1)
    plabels = torch.from_numpy(plabels_np)

    return plabels, centroids

def evaluate_clustering(label, pred, eval_metric=['nmi', 'acc', 'ari'], phase='train'):
    mask = (label != -1)
    label = label[mask]
    pred = pred[mask]
    results = {}
    if 'nmi' in eval_metric:
        nmi = metrics.normalized_mutual_info_score(label, pred, average_method='arithmetic')
        results[f'{phase}_nmi'] = nmi
    if 'ari' in eval_metric:
        ari = metrics.adjusted_rand_score(label, pred)
        results[f'{phase}_ari'] = ari
    if 'f' in eval_metric:
        f = metrics.fowlkes_mallows_score(label, pred)
        results[f'{phase}_f'] = f
    if 'acc' in eval_metric:
        n_clusters = len(set(label))
        if n_clusters == len(set(pred)):
            pred_adjusted = get_y_preds(label, pred, n_clusters=n_clusters)
            acc = metrics.accuracy_score(pred_adjusted, label)
        else:
            acc = 0.
            warnings.warn('TODO: the number of classes is not equal...')
        results[f'{phase}_acc'] = acc
    return results

def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

def compute_cluster_loss(q_centers,
                         k_centers,
                         temperature,
                         psedo_labels,
                         num_cluster):
    # 首先计算当前轮次聚类中心之间的相似度矩阵 d_q
    d_q = q_centers.mm(q_centers.T) / temperature
    # 计算当前轮次聚类中心和历史轮次聚类中心之间的相似度向量 d_k
    d_k = (q_centers * k_centers).sum(dim=1) / temperature
    d_q = d_q.float()
    # 将 d_k 的值分别赋给 d_q 的对角线上的元素，
    # 以确保每个聚类中心与历史轮次中与之对应的聚类中心之间的相似度得到正确计算
    d_q[torch.arange(num_cluster), torch.arange(num_cluster)] = d_k

    # q -> k
    # d_q = q_centers.mm(k_centers.T) / temperature

    # 找出伪标签 psedo_labels 中没有被分配的聚类中心的下标，存储在 zero_classes 中
    zero_classes = torch.arange(num_cluster).cuda()[torch.sum(F.one_hot(torch.unique(psedo_labels),
                                                                             num_cluster), dim=0) == 0]

    # 将没有分配到数据点的聚类中心之间的相似度设置为一个大负数，以便在 softmax 操作中将它们的概率值设为接近于 0 的极小数
    mask = torch.zeros((num_cluster, num_cluster), dtype=torch.bool, device=d_q.device)
    mask[:, zero_classes] = 1
    d_q.masked_fill_(mask, -10)

    # 获取 d_q 矩阵的对角线上的元素，存储在变量 pos 中
    pos = d_q.diag(0)
    mask = torch.ones((num_cluster, num_cluster))
    # 将 mask 张量对角线上的元素全部设置为 0，即生成一个对角线上的元素为 0，其余元素为 1 的矩阵，再转成布尔型
    # 这样生成的 mask 矩阵将在后续的计算中用于掩盖掉 d_q 矩阵中对角线上的元素 pos，从而避免对已经确定的聚类中心进行重新分配
    mask = mask.fill_diagonal_(0).bool()

    # 用于计算聚类中心之间的 softmax 交叉熵损失
    neg = d_q[mask].reshape(-1, num_cluster - 1)
    loss = - pos + torch.logsumexp(torch.cat([pos.reshape(num_cluster, 1), neg], dim=1), dim=1)
    loss[zero_classes] = 0.
    loss = loss.sum() / (num_cluster - len(zero_classes))

    return loss