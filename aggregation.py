import copy
import enum
import torch
import numpy as np
import math
from collections import defaultdict
from hdbscan import HDBSCAN
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp
from scipy.special import rel_entr
import hdbscan
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import *


def get_pca(data, threshold = 0.99):
    normalized_data = StandardScaler().fit_transform(data)
    pca = PCA()
    reduced_data = pca.fit_transform(normalized_data)
    # Determine explained variance using explained_variance_ration_ attribute
    exp_var = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var)
    select_pcas = np.where(cum_sum_eigenvalues <=threshold)[0]
    # print('Number of components with variance <= {:0.0f}%: {}'.format(threshold*100, len(select_pcas)))
    reduced_data = reduced_data[:, select_pcas]
    return reduced_data

eps = np.finfo(float).eps

class LFD():
    def __init__(self, num_classes):
        self.memory = np.zeros([num_classes])

    def clusters_dissimilarity(self, clusters):

        n0 = len(clusters[0])
        n1 = len(clusters[1])
        m = n0 + n1
        cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)
        cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
        mincs0 = np.min(cs0, axis=1)
        mincs1 = np.min(cs1, axis=1)
        ds0 = n0/m * (1 - np.mean(mincs0))
        ds1 = n1/m * (1 - np.mean(mincs1))
        return ds0, ds1

    def aggregate(self, global_model, local_models, ptypes, communication_stats):
        local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
        m = len(local_models)

        # 统计上传通信量
        communication_stats['upload'] += sum(param.numel() for param in local_weights[0].values()) * m * 4  # 假设float32

        for i in range(m):
            local_models[i] = list(local_models[i].parameters())
        global_model = list(global_model.parameters())
        dw = [None for i in range(m)]
        db = [None for i in range(m)]
        for i in range(m):
            dw[i] = global_model[-2].cpu().data.numpy() - local_models[i][-2].cpu().data.numpy()
            db[i] = global_model[-1].cpu().data.numpy() - local_models[i][-1].cpu().data.numpy()
        dw = np.asarray(dw)
        db = np.asarray(db)

        if len(db[0]) <= 2:
            data = [dw[i].reshape(-1) for i in range(m)]
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            labels = kmeans.labels_
            clusters = {0: [], 1: []}
            for i, l in enumerate(labels):
                clusters[l].append(data[i])

            good_cl = 0
            cs0, cs1 = self.clusters_dissimilarity(clusters)
            if cs0 < cs1:
                good_cl = 1

            scores = np.ones([m])
            for i, l in enumerate(labels):
                if l != good_cl:
                    scores[i] = 0

            global_weights = average_weights(local_weights, scores)

            # 统计下载通信量
            communication_stats['download'] += sum(param.numel() for param in global_weights.values()) * 4  # 假设float32

            return global_weights

        # For multiclassification models
        norms = np.linalg.norm(dw, axis=-1)
        self.memory = np.sum(norms, axis=0)
        self.memory += np.sum(abs(db), axis=0)
        max_two_freq_classes = self.memory.argsort()[-2:]
        data = [dw[i][max_two_freq_classes].reshape(-1) for i in range(m)]

        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_
        clusters = {0: [], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(data[i])

        good_cl = 0
        cs0, cs1 = self.clusters_dissimilarity(clusters)
        if cs0 < cs1:
            good_cl = 1

        scores = np.ones([m])
        for i, l in enumerate(labels):
            if l != good_cl:
                scores[i] = 0

        global_weights = average_weights(local_weights, scores)

        # 统计下载通信量
        communication_stats['download'] += sum(param.numel() for param in global_weights.values()) * 4  # 假设float32

        return global_weights




################################################
# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads, communication_stats):
    n_clients = grads.shape[0]

    # 统计上传通信量
    communication_stats['upload'] += grads.size * 4  # 假设 float32 类型，每个参数占用 4 字节

    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)

    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    # 统计下载通信量
    communication_stats['download'] += wv.size * 4  # 假设 float32 类型

    return wv


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers

    def score_gradients(self, local_grads, selected_peers, communication_stats):
        m = len(local_grads)
        grad_len = np.array(local_grads[0][-2].cpu().data.numpy().shape).prod()

        # 统计上传通信量
        communication_stats['upload'] += grad_len * m * 4  # 假设float32

        # 初始化 memory，如果尚未创建
        if self.memory is None:
            self.memory = np.zeros((self.num_peers, grad_len))

        # 累加每个参与者的梯度
        grads = np.zeros((m, grad_len))
        for i in range(m):
            grads[i] = np.reshape(local_grads[i][-2].cpu().data.numpy(), (grad_len))
        self.memory[selected_peers] += grads

        # 计算并获取权重向量
        wv = foolsgold(self.memory)
        self.wv_history.append(wv)

        # 统计下载通信量
        communication_stats['download'] += wv.size * 4  # 假设float32

        return wv[selected_peers]


#######################################################################################
class Tolpegin:
    def __init__(self):
        pass

    def score(self, global_model, local_models, communication_stats,peers_types,selected_peers):
        global_model = list(global_model.parameters())
        last_g = global_model[-2].cpu().data.numpy()
        m = len(local_models)

        # 统计上传通信量
        grad_len = list(local_models[0].parameters())[-2].cpu().data.numpy().size
        communication_stats['upload'] += grad_len * m * 4  # 假设float32

        grads = [None for i in range(m)]
        for i in range(m):
            grad= (last_g - \
                    list(local_models[i].parameters())[-2].cpu().data.numpy())
            grads[i] = grad

        grads = np.array(grads)
        num_classes = grad.shape[0]
        # print('Number of classes:', num_classes)
        dist = [ ]
        labels = [ ]
        for c in range(num_classes):
            data = grads[:, c]
            data = get_pca(copy.deepcopy(data))
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            cl = kmeans.cluster_centers_
            dist.append(((cl[0] - cl[1])**2).sum())
            labels.append(kmeans.labels_)

        dist = np.array(dist)
        candidate_class = dist.argmax()
        print("Candidate source/target class", candidate_class)
        labels = labels[candidate_class]
        if sum(labels) < m/2:
            scores = 1 - labels
        else:
            scores = labels
        # 统计下载通信量
        communication_stats['download'] += scores.size * 4  # 假设float32
        for i, pt in enumerate(peers_types):
            print(pt, 'scored', scores[i])
        return scores
#################################################################################################################
# Clip local updates
def clipp_model(g_w, w, gamma =  1):
    for layer in w.keys():
        w[layer] = g_w[layer] + (w[layer] - g_w[layer])*min(1, gamma)
    return w


def FLAME(global_model, local_models, noise_scalar, communication_stats):
    # Compute number of local models
    m = len(local_models)

    # Flatten local models and compute gradient differences
    g_m = np.array([torch.nn.utils.parameters_to_vector(global_model.parameters()).cpu().data.numpy()])
    f_m = np.array(
        [torch.nn.utils.parameters_to_vector(model.parameters()).cpu().data.numpy() for model in local_models])
    grads = g_m - f_m

    # Update communication stats for uploads (each client uploads its model parameters)
    communication_stats['upload'] += sum(
        param.numel() for param in local_models[0].parameters()) * m * 4  # assuming float32

    # Compute cosine similarity and apply HDBSCAN clustering
    cs = smp.cosine_similarity(grads)
    msc = int(m * 0.5) + 1
    clusterer = hdbscan.HDBSCAN(min_cluster_size=msc, min_samples=1, allow_single_cluster=True)
    clusterer.fit(cs)
    labels = clusterer.labels_

    benign_idxs = np.where(labels != -1)[0] if sum(labels) != -(m) else np.arange(m)

    # Compute distances and clip admitted updates
    euc_d = cdist(g_m, f_m)[0]
    st = np.median(euc_d)
    W_c = []
    for i, idx in enumerate(benign_idxs):
        w_c = clipp_model(global_model.state_dict(), local_models[idx].state_dict(), gamma=st / euc_d[idx])
        W_c.append(w_c)

    # Average admitted clipped updates to obtain new global model
    g_w = average_weights(W_c, np.ones(len(W_c)))

    # Add adaptive noise to the global model
    lamb = 0.001
    sigma = lamb * st * noise_scalar
    for key in g_w.keys():
        noise = torch.FloatTensor(g_w[key].shape).normal_(mean=0, std=(sigma ** 2)).to(g_w[key].device)
        g_w[key] = g_w[key] + noise

    # Update communication stats for download (aggregated global model sent back to clients)
    communication_stats['download'] += sum(param.numel() for param in g_w.values()) * 4  # assuming float32

    return g_w


#################################################################################################################

def median_opt(input,communication_stats):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output


def Repeated_Median_Shard(w, communication_stats):
    SHARD_SIZE = 100000
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)

        # 上传每个模型的权重
        communication_stats['upload'] += total_num * len(w) * 4  # 假设每个值为float32
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        # 下载聚合后的结果
        communication_stats['download'] += total_num * 4
        # 继续执行计算逻辑...


def repeated_median(y, communication_stats):
    num_models = y.shape[1]
    total_num = y.shape[0]

    # 上传 y 的尺寸
    communication_stats['upload'] += total_num * num_models * 4  # 假设 float32 类型
    # 下载中间计算结果
    communication_stats['download'] += total_num * 4
    # 执行中间计算...


# Repeated Median estimator
def Repeated_Median(w, communication_stats):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)

        # 统计上传通信量
        communication_stats['upload'] += total_num * len(w) * 4

        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        # 统计下载通信量
        communication_stats['download'] += total_num * 4
        # 继续执行代码逻辑...


# simple median estimator
def simple_median(w, communication_stats):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)

        # 统计上传通信量
        communication_stats['upload'] += total_num * len(w) * 4

        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        # 统计下载通信量
        communication_stats['download'] += total_num * 4
        # 继续执行代码逻辑...


def trimmed_mean(w, communication_stats,trim_ratio):
    if trim_ratio == 0:
        return average_weights(w, [1 for i in range(len(w))])

    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)

        # 上传每个模型的权重
        communication_stats['upload'] += total_num * len(w) * 4
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        # 下载聚合后的结果
        communication_stats['download'] += total_num * 4
        # 继续执行代码逻辑...


def average_weights(w, marks,communication_stats):
    w_avg = copy.deepcopy(w[0])

    # Step 1: 计算加权平均
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] * (1 / sum(marks))

    communication_stats['upload'] += sum(param.numel() for param in w[0].values()) * len(w) * 4  # 假设float32
    communication_stats['download'] += sum(param.numel() for param in w_avg.values()) * 4  # 聚合后下载
    # if total_w_all_users is not None and epoch is not None:
    #     class_means, class_stds = calculate_mean_std(total_w_all_users)
    # print(f"Epoch {epoch}: Class means: {class_means}")
    # print(f"Epoch {epoch}: Class stds: {class_stds}")

    return w_avg


def Krum(updates,communication_stats=None, f=None , multi=False):
    n = len(updates)
    # 将每个update的参数向量化
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]

    # 统计上传的通信量
    communication_stats['upload'] += sum(param.numel() for param in updates[0]) * n * 4  # 假设每个值是 float32

    updates_ = torch.empty([n, len(updates[0])])
    for i in range(n):
        updates_[i] = updates[i]
    k = n - f - 2

    # 计算点到点距离
    cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k, largest=False)
    dist = dist.sum(1)
    idxs = dist.argsort()

    # 统计下载的通信量
    communication_stats['download'] += sum(param.numel() for param in updates[0]) * 4  # 下载聚合后的结果

    if multi:
        return idxs[:k]
    else:
        return idxs[0]

##################################################################


def NeuDFL(w, marks, avg_w_all, epoch, total_w_all_users, communication_stats, num_users, peers):
    """
    自定义聚合函数：过滤出最可能受到攻击的类别中的表现最差的用户，并对其他用户进行聚合。
    """
    # Step 1: 计算本轮全体用户的平均权重值，找到最可能受到攻击的类别
    attacked_class_idx = np.argmin(avg_w_all)
    print(f"Potentially attacked class in epoch {epoch + 1}: Class {attacked_class_idx}")

    # Step 2: 提取该攻击类别中所有用户的权重值
    attacked_class_w = total_w_all_users[attacked_class_idx, :]

    # 统计上传的通信量（所有用户的上传权重）
    communication_stats['upload'] += sum(param.numel() for param in w[0].values()) * len(w) * 4  # 假设float32

    # Step 3: 计算该类别的均值和方差
    mean_attacked_class_w = np.mean(attacked_class_w)
    std_attacked_class_w = np.std(attacked_class_w)

    threshold = mean_attacked_class_w - 0.5 * std_attacked_class_w  # 可根据需求调整权重
    print(f"Mean w of attacked class: {mean_attacked_class_w}")
    print(f"Std w of attacked class: {std_attacked_class_w}")
    print(f"Threshold for filtering: {threshold}")

    # Step 4: 过滤掉在该攻击类别中表现不佳的用户
    filtered_users_indices = np.where(attacked_class_w < threshold)[0]

    # Step 5: 获取恶意用户ID
    detected_malicious_users = [peers[i].peer_pseudonym for i in filtered_users_indices]
    actual_malicious_users = [peer.peer_pseudonym for peer in peers if peer.peer_type == 'attacker']

    for i in filtered_users_indices:
        marks[i] = 0

    # Step 6: 执行加权聚合
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] * (1 / max(1, sum(marks)))

    # 统计下载的通信量（聚合后下载）
    communication_stats['download'] += sum(param.numel() for param in w_avg.values()) * 4  # 聚合后的模型下载量

    return w_avg, detected_malicious_users, actual_malicious_users, attacked_class_idx

