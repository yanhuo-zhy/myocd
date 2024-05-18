import argparse
import protopformer_slc as protopformer
import torch
from torchvision import transforms
from copy import deepcopy
import pickle
from tools.CD_cub import get_cub_datasets
from tools.CD_cars import get_scars_datasets
from tools.CD_food101 import get_food_101_datasets
from tools.CD_pets import get_oxford_pets_datasets
from tools.CD_inaturalist import get_inaturalist_datasets
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


#baslien 36
def sequential_leader_clustering_with_prior_new(input_data, threshold_coefficient, class_means, max_distances, processed_data, targets_data):
    # 打印输入数据的统计摘要
    # print("Input Data Summary:")
    # print(f"input_data: Max: {input_data.max()}, Min: {input_data.min()}, Mean: {input_data.mean()}")
    # print(f"class_means: Max: {class_means.max()}, Min: {class_means.min()}, Mean: {class_means.mean()}")
    # print(f"max_distances: Max: {max_distances.max()}, Min: {max_distances.min()}, Mean: {max_distances.mean()}")
    # print(f"processed_data: Max: {processed_data.max()}, Min: {processed_data.min()}, Mean: {processed_data.mean()}")
    # print(f"targets_data: Max: {targets_data.max()}, Min: {targets_data.min()}, Mean: {targets_data.float().mean()}")


    # 更新预设的阈值
    thresholds = threshold_coefficient * max_distances
    my_threshold = thresholds.mean().item()
    # my_threshold = torch.full_like(thresholds, thresholds.max().item())
    new_targets = []

    for data_point in input_data:
        # 计算该点与所有现有簇平均向量之间的距离
        distances = torch.norm(class_means - data_point, dim=1)

        # 检查是否存在距离小于阈值的簇
        within_threshold = distances < my_threshold
        if within_threshold.any():
            # 归入最近的簇，且该簇的距离小于阈值
            min_distances_within_threshold = distances[within_threshold]
            nearest_cluster_index = torch.argmin(min_distances_within_threshold)
            cluster_id = torch.nonzero(within_threshold)[nearest_cluster_index].item()
            # nearest_cluster = torch.argmin(distances)
            # cluster_id = nearest_cluster.item()
        else:
            # 创建新的簇
            cluster_id = class_means.size(0)
            class_means = torch.cat((class_means, data_point.unsqueeze(0)), dim=0)
            # 初始化新簇的最大距离和阈值
            # max_distances = torch.cat((max_distances, torch.tensor([0])))
            # new_threshold = thresholds.mean().item()
            # thresholds = torch.cat((thresholds, torch.tensor([new_threshold])))
            # my_threshold = torch.cat((my_threshold, torch.tensor([new_threshold])))

        # 更新类别标签
        new_targets.append(cluster_id)

        # 更新处理过的数据和类别信息
        processed_data = torch.cat((processed_data, data_point.unsqueeze(0)), dim=0)
        targets_data = torch.cat((targets_data, torch.tensor([cluster_id])), dim=0)

        # 更新该类别的平均向量
        class_activations = processed_data[targets_data == cluster_id]
        class_means[cluster_id] = class_activations.mean(dim=0)
        
        # # 更新最大距离和阈值
        # current_max_distance = torch.norm(class_activations - class_means[cluster_id], dim=1).max()
        # if current_max_distance > max_distances[cluster_id]:
        #     max_distances[cluster_id] = current_max_distance
        #     thresholds[cluster_id] = threshold_coefficient * current_max_distance

    return torch.tensor(new_targets)  # 只返回处理过的数据点的类别标签

def compute_class_means_and_max_distances(activations, targets, num_classes=100):
    class_means = []
    max_distance = []

    for class_id in range(num_classes):
        # 提取属于当前类别的向量
        class_activations = activations[targets == class_id]

        # 计算类别的平均向量
        mean_vector = class_activations.mean(dim=0)
        class_means.append(mean_vector)

        # 计算该类别中所有向量与平均向量之间的距离
        distances = torch.norm(class_activations - mean_vector, dim=1)

        # 计算最大距离
        max_distance_class = distances.max()
        max_distance.append(max_distance_class)

    return torch.stack(class_means), torch.tensor(max_distance)

def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc

def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


# 设置命令行参数
parser = argparse.ArgumentParser(description='Run model inference with given model path and dataset name.')
parser.add_argument('--model_path', type=str, help='Path to the model file.')
parser.add_argument('--dataset', type=str, default='cub', help='Name of the dataset (e.g., "cub").')
args = parser.parse_args()

if args.dataset == 'CD_CUB2011U':
    labeled_nums=100

elif args.dataset == 'CD_Car':
    labeled_nums=98

elif args.dataset == 'CD_food':
    labeled_nums=51
    
elif args.dataset == 'CD_pets':
    labeled_nums=19

elif args.dataset == 'Animalia':
    labeled_nums=39
    
elif args.dataset == 'Arachnida':
    labeled_nums=28
    
elif args.dataset == 'Fungi':
    labeled_nums=61
    
elif args.dataset == 'Mollusca':
    labeled_nums=47

# 载入模型
model = protopformer.construct_PPNet_dino(base_architecture="deit_base_patch16_224",
                            pretrained=True, img_size=224,
                            prototype_shape=[labeled_nums*5, 768, 1, 1],
                            num_classes=labeled_nums,
                            reserve_layers=11,
                            reserve_token_nums=196,
                            use_global=True,
                            use_ppc_loss=False,
                            ppc_cov_thresh=1.,
                            ppc_mean_thresh=2.,
                            global_coe=0.5,
                            global_proto_per_class=5,
                            prototype_activation_function="log",
                            add_on_layers_type="regular")

# 模型加载代码部分
state_dict = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
# 载入数据集
if args.dataset == 'CD_CUB2011U':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
    transforms.Resize(int(224 / 0.875), interpolation=3),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    split_path = 'ssb_splits/cub_osr_splits.pkl'
    with open(split_path, 'rb') as handle:
        class_info = pickle.load(handle)

    train_classes = class_info['known_classes']
    open_set_classes = class_info['unknown_classes']
    unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

    train_dataset, test_dataset, train_dataset_unlabelled = get_cub_datasets(train_transform=transform, test_transform=test_transform, 
                                train_classes=train_classes, prop_train_labels=0.5)
    print("train_classes:", train_classes)
    print("len(train_classes):", len(train_classes))
    print("unlabeled_classes:", unlabeled_classes)
    print("len(unlabeled_classes):", len(unlabeled_classes))
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(train_classes) + list(unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    train_dataset.target_transform = target_transform
    test_dataset.target_transform = target_transform
    train_dataset_unlabelled.target_transform = target_transform

elif args.dataset == 'CD_Car':

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
    transforms.Resize(int(224 / 0.875), interpolation=3),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    split_path = 'ssb_splits/scars_osr_splits.pkl'
    with open(split_path, 'rb') as handle:
        class_info = pickle.load(handle)

    train_classes = class_info['known_classes']
    open_set_classes = class_info['unknown_classes']
    unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

    train_dataset, test_dataset, train_dataset_unlabelled = get_scars_datasets(train_transform=transform, test_transform=test_transform, 
                                train_classes=train_classes, prop_train_labels=0.5)
    print("train_classes:", train_classes)
    print("len(train_classes):", len(train_classes))
    print("unlabeled_classes:", unlabeled_classes)
    print("len(unlabeled_classes):", len(unlabeled_classes))
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(train_classes) + list(unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    train_dataset.target_transform = target_transform
    test_dataset.target_transform = target_transform
    train_dataset_unlabelled.target_transform = target_transform

elif args.dataset == 'CD_food':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
    transforms.Resize(int(224 / 0.875), interpolation=3),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    train_dataset, test_dataset, train_dataset_unlabelled = get_food_101_datasets(train_transform=transform, test_transform=test_transform, 
                                train_classes=range(51), prop_train_labels=0.5)
    
elif args.dataset == 'CD_pets':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
    transforms.Resize(int(224 / 0.875), interpolation=3),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    train_dataset, test_dataset, train_dataset_unlabelled = get_oxford_pets_datasets(train_transform=transform, test_transform=test_transform, 
                                train_classes=range(19), prop_train_labels=0.5)

elif args.dataset == 'Animalia':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
    transforms.Resize(int(224 / 0.875), interpolation=3),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Animalia',
                                train_classes=range(39), prop_train_labels=0.5)
    
elif args.dataset == 'Arachnida':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
    transforms.Resize(int(224 / 0.875), interpolation=3),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Arachnida',
                                train_classes=range(28), prop_train_labels=0.5)
    
elif args.dataset == 'Fungi':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
    transforms.Resize(int(224 / 0.875), interpolation=3),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Fungi',
                                train_classes=range(61), prop_train_labels=0.5)
    
elif args.dataset == 'Mollusca':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
    transforms.Resize(int(224 / 0.875), interpolation=3),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])
    # transform = ContrastiveLearningViewGenerator(base_transform=transform, n_views=2)

    train_dataset, test_dataset, train_dataset_unlabelled = get_inaturalist_datasets(train_transform=transform, test_transform=test_transform, subclassname='Mollusca',
                                train_classes=range(47), prop_train_labels=0.5)


unlabelled_train_examples_test = deepcopy(train_dataset_unlabelled)
unlabelled_train_examples_test.transform = test_transform
train_dataset.transform = test_transform

train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=8,
                                    batch_size=256, shuffle=False, pin_memory=False)

test_loader_unlabelled = torch.utils.data.DataLoader(unlabelled_train_examples_test, num_workers=8,
                                    batch_size=256, shuffle=False, pin_memory=False)


# 得到train_feat & targets

model.to('cuda')
model.eval()

all_train_feats = []
train_targets = []

for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader)):
    with torch.no_grad():
        images = images.cuda()
        label = label.cuda()
        _, _, _, feats = model(images)

    all_train_feats.append(feats.cpu())
    train_targets.append(label.cpu())

all_train_feats = torch.cat(all_train_feats, dim=0)
train_targets = torch.cat(train_targets, dim=0)


# 得到test_feat & targets

model.to('cuda')
model.eval()

all_test_feats = []
test_targets = []

for batch_idx, (images, label, _, _) in enumerate(tqdm(test_loader_unlabelled)):
    with torch.no_grad():
        images = images.cuda()
        label = label.cuda()
        _, _, _, feats = model(images)

    all_test_feats.append(feats.cpu())
    test_targets.append(label.cpu())

all_test_feats = torch.cat(all_test_feats, dim=0)
test_targets = torch.cat(test_targets, dim=0)

print("all_train_feats.shape", all_train_feats.shape)
mean_vectors, max_dis = compute_class_means_and_max_distances(all_train_feats, train_targets, num_classes=labeled_nums)
print("mean_vectors.shape:",mean_vectors.shape)
print("max_dis.shape:",max_dis.shape)
print("max_dis.max:",max_dis.max())

input_data = all_test_feats
threshold_coefficient=1.2
class_means=mean_vectors
max_distances=max_dis
processed_data=all_train_feats
targets_data=train_targets

new_targets = sequential_leader_clustering_with_prior_new(
    input_data=input_data,
    threshold_coefficient=3.0,
    class_means=class_means,
    max_distances=max_distances,
    processed_data=processed_data,
    targets_data=targets_data)

targets = test_targets.numpy()


mask = (targets < labeled_nums)
total_acc, old_acc, new_acc = split_cluster_acc_v1(targets, new_targets.cpu().numpy(), mask)
print("ACC V1")
print(f"total_acc: {total_acc}")
print(f"old_acc: {old_acc}")
print(f"new_acc: {new_acc}")
total_acc, old_acc, new_acc = split_cluster_acc_v2(targets, new_targets.cpu().numpy(), mask)
print("ACC V2")
print(f"total_acc: {total_acc}")
print(f"old_acc: {old_acc}")
print(f"new_acc: {new_acc}")