"""
k-NN 分类器 (k-NN classifier)

基于 faiss 的 k 近邻分类实现，用于冻结特征上的 k-NN 评测
（论文中 SAR-VSA / FUSAR-Ship 的分类评测方式）。

A faiss-based k-NN classifier for evaluating frozen features
(the protocol used for SAR-VSA / FUSAR-Ship in the paper).
"""

import numpy as np
import torch
import faiss


class KNNClassifier:
    """k-NN 分类器 (k-NN classifier).

    Args:
        k_values: 需要评测的 k 值列表 (list of k values to evaluate)
        temperature: 投票 softmax 的温度系数 (temperature for softmax voting)
        metric: 'cosine'（余弦相似度）或 'l2'
        skip_first_nn: 是否跳过最近邻（用于排除查询样本本身）(skip the first nearest neighbor)
    """

    def __init__(self, k_values=(1, 5, 10, 20), temperature=0.07,
                 metric='cosine', skip_first_nn=False):
        self.k_values = k_values
        self.temperature = temperature
        self.metric = metric
        self.skip_first_nn = skip_first_nn
        self.index = None
        self.train_labels = None

    def fit(self, features, labels):
        """用训练特征构建检索索引 (Build the retrieval index from training features)."""
        features_np = features.numpy().astype('float32')

        if self.metric == 'cosine':
            faiss.normalize_L2(features_np)

        d = features_np.shape[1]
        if self.metric == 'cosine':
            self.index = faiss.IndexFlatIP(d)   # 内积 = 余弦相似度 (inner product = cosine similarity)
        else:
            self.index = faiss.IndexFlatL2(d)

        self.index.add(features_np)
        self.train_labels = labels.numpy()

    def predict(self, features, k=None):
        """预测类别得分 (Predict class scores)."""
        if k is None:
            k = max(self.k_values)

        features_np = features.numpy().astype('float32')
        if self.metric == 'cosine':
            faiss.normalize_L2(features_np)

        k_search = k + (1 if self.skip_first_nn else 0)
        D, I = self.index.search(features_np, k_search)  # D: 相似度/距离, I: 近邻索引

        if self.skip_first_nn:
            D = D[:, 1:]
            I = I[:, 1:]

        D = torch.from_numpy(D[:, :k])
        I = torch.from_numpy(I[:, :k])

        # 加权投票 (weighted voting)
        neighbor_labels = torch.from_numpy(self.train_labels[I.numpy()])
        weights = torch.nn.functional.softmax(D / self.temperature, dim=1)

        num_classes = len(np.unique(self.train_labels))
        scores = torch.zeros(features.shape[0], num_classes)
        for i in range(features.shape[0]):
            for j in range(k):
                label = neighbor_labels[i, j]
                scores[i, label] += weights[i, j]
        return scores

    def evaluate(self, test_features, test_labels):
        """评测不同 k 值的 Top-1 / Top-5 准确率。

        Evaluate Top-1 / Top-5 accuracy for each k value.
        """
        results = {}
        for k in self.k_values:
            scores = self.predict(test_features, k)
            predictions = torch.argmax(scores, dim=1)
            accuracy = (predictions == test_labels).float().mean().item() * 100
            results[f'Top-1 (k={k})'] = accuracy

            if scores.shape[1] >= 5:
                _, top5_pred = scores.topk(5, dim=1)
                top5_correct = torch.any(top5_pred == test_labels.unsqueeze(1), dim=1)
                top5_accuracy = top5_correct.float().mean().item() * 100
                results[f'Top-5 (k={k})'] = top5_accuracy
        return results
