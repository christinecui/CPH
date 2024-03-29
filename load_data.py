import torch
import scipy.io as sio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class CustomDataSet(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index], index

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.labels)
        return count

def get_loader_source(batch_size, base_path, domain_name):
    path = base_path + domain_name + '_feature.mat'

    # 读取.mat文件
    data = sio.loadmat(path)

    # 将数据和标签分别存储在不同的变量中
    data_tensor = torch.from_numpy(data['deepfea'])
    label_tensor = torch.from_numpy(data['label'])

    if base_path == '/datasets/OfficeHome_mat/':
        label_tensor = label_tensor.T

    # 创建CustomDataSet对象
    source_dataset = CustomDataSet(images=data_tensor, labels=label_tensor)

    # 创建DataLoader对象
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,  # 每个epochs乱序
                                                num_workers=4)
    # 得到类别数量
    classes = torch.unique(label_tensor)
    n_class = classes.size(0)

    # 得到特征维度
    dim_fea = data_tensor.size(1)

    return source_loader, n_class, dim_fea

def get_loader_target(batch_size, base_path, domain_name):
    path = base_path + domain_name + '_feature.mat'

    # 读取.mat文件
    data = sio.loadmat(path)

    # x_mean = np.mean(train_x, axis=0).reshape((1, -1))
    # train_x = train_x - x_mean

    # 将数据和标签分别存储在不同的变量中
    data_tensor = torch.from_numpy(data['deepfea'])
    label_tensor = torch.from_numpy(data['label'])
    # label_tensor = torch.from_numpy(data['label']).squeeze().unsqueeze(dim=1)

    if base_path == '/datasets/OfficeHome_mat/':
        label_tensor = label_tensor.T

    # 创建训练集和测试集
    train_data, test_data, train_label, test_label = train_test_split(data_tensor,
                                                                      label_tensor,
                                                                      test_size=0.1,
                                                                      random_state=42)
    # # 打印训练集和测试集的形状
    # print('Training data shape:', train_data.shape)
    # print('Testing data shape:', test_data.shape)
    # print('Training label shape:', train_label.shape)
    # print('Testing label shape:', test_label.shape)

    imgs = {'train': train_data, 'query': test_data}
    labels = {'train': train_label, 'query': test_label}

    dataset = {x: CustomDataSet(images=imgs[x], labels=labels[x])
               for x in ['train', 'query']}

    shuffle = {'train': True, 'query': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in ['train', 'query']}

    return dataloader



