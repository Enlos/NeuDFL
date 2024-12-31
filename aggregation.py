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

    def aggregate(self, global_model, local_models, ptypes):
        local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
        m = len(local_models)

        # 统计上传通信量
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
        return global_weights




################################################
# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]


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
    return wv


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers

    def score_gradients(self, local_grads, selected_peers):
        m = len(local_grads)
        grad_len = np.array(local_grads[0][-2].cpu().data.numpy().shape).prod()


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

        return wv[selected_peers]


#######################################################################################
class Tolpegin:
    def __init__(self):
        pass

    def score(self, global_model, local_models,peers_types,selected_peers):
        global_model = list(global_model.parameters())
        last_g = global_model[-2].cpu().data.numpy()
        m = len(local_models)

        # 统计上传通信量
        grad_len = list(local_models[0].parameters())[-2].cpu().data.numpy().size

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
        for i, pt in enumerate(peers_types):
            print(pt, 'scored', scores[i])
        return scores
#################################################################################################################
# Clip local updates
def clipp_model(g_w, w, gamma =  1):
    for layer in w.keys():
        w[layer] = g_w[layer] + (w[layer] - g_w[layer])*min(1, gamma)
    return w


def FLAME(global_model, local_models, noise_scalar):
    # Compute number of local models
    m = len(local_models)

    # Flatten local models and compute gradient differences
    g_m = np.array([torch.nn.utils.parameters_to_vector(global_model.parameters()).cpu().data.numpy()])
    f_m = np.array(
        [torch.nn.utils.parameters_to_vector(model.parameters()).cpu().data.numpy() for model in local_models])
    grads = g_m - f_m

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
    return g_w


#################################################################################################################
def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output

def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y
    return w_med


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


# Repeated Median estimator
def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med

# simple median estimator
def simple_median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    return w_med

def trimmed_mean(w, trim_ratio):
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
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    return w_med


# Get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] *(1/sum(marks))
    return w_avg


def Krum(updates, f=None , multi=False):
    n = len(updates)
    # 将每个update的参数向量化
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]

    updates_ = torch.empty([n, len(updates[0])])
    for i in range(n):
        updates_[i] = updates[i]
    k = n - f - 2

    # 计算点到点距离
    cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k, largest=False)
    dist = dist.sum(1)
    idxs = dist.argsort()

    if multi:
        return idxs[:k]
    else:
        return idxs[0]

##################################################################
def NeuDFL(w, marks, epoch, k, num_users, peers):
    """
    服务器端自定义聚合函数：过滤出最可能受到攻击的类别中的表现最差的用户，并对其他用户进行聚合。
    """
    # Step 1: 服务器提取每个用户的 FCL 权重，并按类别展开求和
    total_w_all_users = np.zeros((len(w[0]['fc.weight']), num_users))  # 假设 fc.weight 是 FCL 权重
    for idx, peer_w in enumerate(w):
        fc_weight = peer_w['fc.weight'].cpu().numpy()  # 提取全连接层权重
        total_w_all_users[:, idx] = np.sum(fc_weight, axis=1)  # 按类别求和

    # Step 2: 计算本轮全体用户的平均权重值，找到最可能受到攻击的类别
    avg_w_all = np.mean(total_w_all_users, axis=1)
    attacked_class_idx = np.argmin(avg_w_all)
    print(f"Potentially attacked class in epoch {epoch + 1}: Class {attacked_class_idx}")

    # Step 3: 提取该攻击类别中所有用户的权重值
    attacked_class_w = total_w_all_users[attacked_class_idx, :]
    mean_attacked_class_w = np.mean(attacked_class_w)
    std_attacked_class_w = np.std(attacked_class_w)
    threshold = mean_attacked_class_w - k * std_attacked_class_w  # 可根据需求调整权重
    print(f"Hyperparameter K: {k}")
    print(f"Mean w of attacked class: {mean_attacked_class_w}")
    print(f"Std w of attacked class: {std_attacked_class_w}")
    print(f"Threshold for filtering: {threshold}")

    # Step 4: 过滤掉在该攻击类别中表现不佳的用户
    filtered_users_indices = np.where(attacked_class_w < threshold)[0]
    for i in filtered_users_indices:
        marks[i] = 0

    # Step 5: 获取恶意用户ID
    detected_malicious_users = [peers[i].peer_pseudonym for i in filtered_users_indices]
    actual_malicious_users = [peer.peer_pseudonym for peer in peers if peer.peer_type == 'attacker']

    # Step 6: 执行加权聚合
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] * (1 / max(1, sum(marks)))

    # 返回结果
    return w_avg, detected_malicious_users, actual_malicious_users, attacked_class_idx


# def NeuDFL(w, marks, avg_w_all, epoch, total_w_all_users,k , num_users, peers):
#     """
#     自定义聚合函数：过滤出最可能受到攻击的类别中的表现最差的用户，并对其他用户进行聚合。
#     """
#     # Step 1: 计算本轮全体用户的平均权重值，找到最可能受到攻击的类别
#     attacked_class_idx = np.argmin(avg_w_all)
#     print(f"Potentially attacked class in epoch {epoch + 1}: Class {attacked_class_idx}")
#
#     # Step 2: 提取该攻击类别中所有用户的权重值
#     attacked_class_w = total_w_all_users[attacked_class_idx, :]
#
#     # 统计上传的通信量（所有用户的上传权重）
#     # Step 3: 计算该类别的均值和方差
#     mean_attacked_class_w = np.mean(attacked_class_w)
#     std_attacked_class_w = np.std(attacked_class_w)
#     threshold = mean_attacked_class_w - k * std_attacked_class_w  # 可根据需求调整权重
#     print(f"hypercharacter K : {k}")
#     print(f"Mean w of attacked class: {mean_attacked_class_w}")
#     print(f"Std w of attacked class: {std_attacked_class_w}")
#     print(f"Threshold for filtering: {threshold}")
#
#     # Step 4: 过滤掉在该攻击类别中表现不佳的用户
#     filtered_users_indices = np.where(attacked_class_w < threshold)[0]
#
#     # Step 5: 获取恶意用户ID
#     detected_malicious_users = [peers[i].peer_pseudonym for i in filtered_users_indices]
#     actual_malicious_users = [peer.peer_pseudonym for peer in peers if peer.peer_type == 'attacker']
#
#     for i in filtered_users_indices:
#         marks[i] = 0
#
#     # Step 6: 执行加权聚合
#     w_avg = copy.deepcopy(w[0])
#     for key in w_avg.keys():
#         w_avg[key] = w_avg[key] * marks[0]
#     for key in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[key] += w[i][key] * marks[i]
#         w_avg[key] = w_avg[key] * (1 / max(1, sum(marks)))
#
#     # 统计下载的通信量（聚合后下载）
#     return w_avg, detected_malicious_users, actual_malicious_users, attacked_class_idx

