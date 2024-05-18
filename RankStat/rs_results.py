import argparse
import protopformer_rs as protopformer
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
                            prototype_shape=[labeled_nums*5, 32, 1, 1],
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


test_loader_unlabelled = torch.utils.data.DataLoader(unlabelled_train_examples_test, num_workers=8,
                                    batch_size=256, shuffle=False, pin_memory=False)


# 得到所有特征

model.to('cuda')
model.eval()

all_feats = []
targets = np.array([])

for batch_idx, (images, label, _, _) in enumerate(tqdm(test_loader_unlabelled)):
    with torch.no_grad():
        images = images.cuda()
        label = label.cuda()
        _, _, _, feats = model(images)

    all_feats.append(feats.cpu().numpy())
    targets = np.append(targets, label.cpu().numpy())

all_feats = np.concatenate(all_feats, axis=0)

top_3_indices = np.argsort(-all_feats, axis=1)[:, :3]
sorted_top_3_indices = np.sort(top_3_indices, axis=1).tolist()


preds = []
hash_dict = []
for feat in sorted_top_3_indices:
    if not feat in hash_dict:
        hash_dict.append(feat)
    preds.append(hash_dict.index(feat))
preds = np.array(preds)


mask = (targets < labeled_nums)
total_acc, old_acc, new_acc = split_cluster_acc_v1(targets, preds, mask)
print("ACC V1")
print(f"total_acc: {total_acc}")
print(f"old_acc: {old_acc}")
print(f"new_acc: {new_acc}")
total_acc, old_acc, new_acc = split_cluster_acc_v2(targets, preds, mask)
print("ACC V2")
print(f"total_acc: {total_acc}")
print(f"old_acc: {old_acc}")
print(f"new_acc: {new_acc}")