# code ref: TrajGAT - https://github.com/HuHaonan-CHN/TrajGAT

import copy
import logging
import numpy as np
import scipy.sparse as sp
import torch
import dgl
import logging


def laplacian_positional_encoding(g, adj, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = adj.tocsr().astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    L = L.toarray()

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    
    # if temp.shape[1] != pos_enc_dim:
    #     print("!!!! ERROR", temp.shape)
    g.ndata["lap_pos_feat"] = torch.from_numpy(EigVec[:, 1 : pos_enc_dim + 1]).float()
    return g

def laplacian_positional_encoding_dense(g, adj, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        much faster than original laplacian_positional_encoding -- yc
    """
    A = adj.toarray().astype(float)
    N = np.zeros(A.shape)
    np.fill_diagonal(N, np.sum(A>0, axis = 0).clip(1) ** -0.5)
    L = np.eye(g.number_of_nodes()) - np.matmul(np.matmul(N, A), N)
    
    # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    
    # if temp.shape[1] != pos_enc_dim:
    #     print("!!!! ERROR", temp.shape)
    g.ndata["lap_pos_feat"] = torch.from_numpy(EigVec[:, 1 : pos_enc_dim + 1]).float()
    return g


def trajlist_to_trajgraph(traj_list_list, qtree, qtree_name2id, lon_range, lat_range):
    trajdict_list_list = _prepare(traj_list_list, qtree, qtree_name2id, lon_range, lat_range)
    trajgraph_list_list = _build_graph(trajdict_list_list)
    return trajgraph_list_list

        
def _prepare(traj_l, qtree, qtree_name2id, lon_range, lat_range) :
    # trajdict_list_list = []
    # for traj_l in traj_l_l:
    #     temp_list = []
    #     for traj in traj_l:
    #         padding_traj = []
    #         point_ids = []
    #         adj, add_feat, tree_id = _build_adj_matrix_tree(traj, qtree, qtree_name2id)
    #         padding_traj.extend([(tp[0], tp[1], 0, 0) for tp in traj])
    #         padding_traj.extend(add_feat)

    #         point_ids.extend([0 for _ in range(len(traj))])
    #         point_ids.extend(tree_id)

    #         # 用来标注 真实轨迹点 和 虚拟节点的 feature , 1: truth node    0: visual node
    #         flag = torch.zeros((len(padding_traj), 1))
    #         flag[0 : len(traj)] = 1

    #         temp_list.append({"traj": _normalize(padding_traj, lon_range, lat_range), "adj": adj, "flag": flag, "point_ids": torch.tensor(point_ids).long()})
    #     trajdict_list_list.append(temp_list)

    # return trajdict_list_list
    rtn = []
    for traj in traj_l:
        padding_traj = []
        point_ids = []
        adj, add_feat, tree_id = _build_adj_matrix_tree(traj, qtree, qtree_name2id)
        padding_traj.extend([(tp[0], tp[1], 0, 0) for tp in traj])
        padding_traj.extend(add_feat)

        point_ids.extend([0 for _ in range(len(traj))])
        point_ids.extend(tree_id)

        # 用来标注 真实轨迹点 和 虚拟节点的 feature , 1: truth node    0: visual node
        flag = torch.zeros((len(padding_traj), 1))
        flag[0 : len(traj)] = 1

        rtn.append({"traj": _normalize(padding_traj, lon_range, lat_range), "adj": adj, "flag": flag, "point_ids": torch.tensor(point_ids).long()})
    return rtn


def _normalize(traj, lon_range, lat_range):
    # lon_mean, lon_std, lat_mean, lat_std = self.data_features
    traj = torch.tensor(traj)
    if traj.shape[1] == 2:
        # traj = traj - torch.tensor([lon_mean, lat_mean])
        # traj = traj * torch.tensor([1 / lon_std, 1 / lat_std])
        traj = traj - torch.tensor([lon_range[0], lat_range[0]])
        traj = traj / torch.tensor([ (lon_range[1] - lon_range[0])/2 , (lat_range[1] - lat_range[0])/2])
        traj = traj - 1
    elif traj.shape[1] == 4:
        # traj = traj - torch.tensor([lon_mean, lat_mean, 0, 0])
        # traj = traj * torch.tensor([1 / lon_std, 1 / lat_std, 1, 1])
        traj = traj - torch.tensor([lon_range[0], lat_range[0], 0, 0])
        traj = traj / torch.tensor([ (lon_range[1] - lon_range[0])/2 , (lat_range[1] - lat_range[0])/2, 1, 1])
        traj = traj - 1
    return traj


def _build_adj_matrix_tree(traj, qtree, qtree_name2id, vir_node_layers=1):
    if vir_node_layers == 1:
        # 根据qtree结构，构建涉及 vir_node_layer 层父节点的全连接graph
        traj_len = len(traj)
        point2treel = []  # 每个点在qtree中对应的 vir_node_layer 个父节点

        for t_point in traj:
            t_list = qtree.intersect(t_point, method="tree")
            point2treel.append(t_list)

        node_num = traj_len
        tree_set = []
        for treel in point2treel:
            tree_set.extend(treel)
        tree_set = set(tree_set)
        node_num += len(tree_set)

        id_start = traj_len
        tree2id = {}
        center_wh_feat = []
        tree_id = []
        for tt in tree_set:
            tree2id[hash(tt)] = id_start
            id_start += 1
            this_x, this_y = tt.center
            this_w, this_h = tt.width, tt.height
            center_wh_feat.append((this_x, this_y, this_w, this_h))  # 将格子 中心点坐标 宽 高 作为他的特征

            if qtree_name2id:
                # 将每个结点对应的id进行存储
                tree_id.append(qtree_name2id[hash(tt)])
            else:
                # 不用word embedding的时候，补0
                tree_id.append(0)

        u = []
        v = []
        edge_data = []

        #  连接图上纵向的边
        for point_index in range(traj_len):
            tree_list = point2treel[point_index]
            # 连接 point —— 叶子树
            for tt in tree_list:
                u.append(point_index)
                v.append(tree2id[hash(tt)])
                edge_data.append(1)

                u.append(tree2id[hash(tt)])
                v.append(point_index)
                edge_data.append(1)

        # 连接图上横向的边
        tree_ids = list(tree2id.values())
        for i in range(len(tree_ids) - 1):
            for j in range(i + 1, len(tree_ids)):
                u.append(tree_ids[i])
                v.append(tree_ids[j])
                edge_data.append(1)

                u.append(tree_ids[j])
                v.append(tree_ids[i])
                edge_data.append(1)

        # 自身的连边
        for this_id in range(node_num):
            u.append(this_id)
            v.append(this_id)
            edge_data.append(1)

        u = np.array(u)
        v = np.array(v)
        edge_data = np.array(edge_data)
        adj_matrix = sp.coo_matrix((edge_data, (u, v)), shape=(node_num, node_num))
    else:
        # 根据qtree结构，构建涉及 vir_node_layer 层父节点的全连接graph
        traj_len = len(traj)
        point2treel = []  # 每个点在qtree中对应的 vir_node_layer 个父节点

        for t_point in traj:
            t_list = qtree.intersect(t_point, method="all_tree")
            point2treel.append([i[1] for i in t_list[-1 : -1 - vir_node_layers : -1]])

        node_num = traj_len
        tree_set = []
        for treel in point2treel:
            tree_set.extend(treel)
        tree_set = set(tree_set)
        node_num += len(tree_set)

        id_start = traj_len
        tree2id = {}
        center_wh_feat = []
        tree_id = []
        for tt in tree_set:
            tree2id[hash(tt)] = id_start
            id_start += 1
            this_x, this_y = tt.center
            this_w, this_h = tt.width, tt.height
            center_wh_feat.append((this_x, this_y, this_w, this_h))  # 将格子 中心点坐标 宽 高 作为他的特征

            if qtree_name2id:
                # 将每个结点对应的id进行存储
                tree_id.append(qtree_name2id[hash(tt)])
            else:
                # 不用word embedding的时候，补0
                tree_id.append(0)

        u = []
        v = []
        edge_data = []

        #  连接图上纵向的边
        for point_index in range(traj_len):
            tree_list = point2treel[point_index]
            # 连接 point —— 叶子树
            u.append(tree2id[hash(tree_list[0])])
            v.append(point_index)
            edge_data.append(1)

            v.append(tree2id[hash(tree_list[0])])
            u.append(point_index)
            edge_data.append(1)

            # 连接 叶子树 —— 更高层的树
            for tt in range(1, len(tree_list)):
                u.append(tree2id[hash(tree_list[tt])])
                v.append(tree2id[hash(tree_list[tt - 1])])
                edge_data.append(1)

                u.append(tree2id[hash(tree_list[tt - 1])])
                v.append(tree2id[hash(tree_list[tt])])
                edge_data.append(1)

        # 连接图上横向的边
        tree_ids = set()
        for jj in range(traj_len):
            tree_ids.add(point2treel[jj][-1])
        tree_ids = list(tree_ids)

        for i in range(len(tree_ids) - 1):
            for j in range(i + 1, len(tree_ids)):
                u.append(tree2id[hash(tree_ids[i])])
                v.append(tree2id[hash(tree_ids[j])])
                edge_data.append(1)

                u.append(tree2id[hash(tree_ids[j])])
                v.append(tree2id[hash(tree_ids[i])])
                edge_data.append(1)

        # 自身的连边
        for this_id in range(node_num):
            u.append(this_id)
            v.append(this_id)
            edge_data.append(1)

        u = np.array(u)
        v = np.array(v)
        edge_data = np.array(edge_data)
        try:
            adj_matrix = sp.coo_matrix((edge_data, (u, v)), shape=(node_num, node_num))
        except:
            logging.info("edge_data:\n{}".format(edge_data))
            logging.info("U\n".format(u))
            logging.info("V\n".format(v))
            logging.info("NN\n".format(node_num))
            logging.info("FINISH")
    return adj_matrix, center_wh_feat, tree_id


def _build_graph(trajdict_l, d_lap_pos = 8):
    # trajgraph_list_list = []

    # for trajdict_l in trajdict_l_l:
    #     trajgraph_l = []
    #     for trajdict in trajdict_l:
    #         node_features = trajdict["traj"].float()
    #         id_features = trajdict["point_ids"]
    #         adj = trajdict["adj"]
    #         flag = trajdict["flag"]

    #         # Create the DGL Graph
    #         g = dgl.from_scipy(adj, eweight_name="feat")

    #         padding_node_num = g.num_nodes() - node_features.shape[0]
    #         padding_node = torch.zeros((padding_node_num, node_features.shape[1])).float()
    #         all_node_features = torch.cat((node_features, padding_node), dim=0)

    #         g.ndata["feat"] = all_node_features
    #         g.ndata["flag"] = flag
    #         g.ndata["hash"] = id_features

    #         g = laplacian_positional_encoding(g, d_lap_pos)

    #         trajgraph_l.append(g)

    #     trajgraph_list_list.append(copy.deepcopy(trajgraph_l))

    # return trajgraph_list_list

    rtn = []

    for trajdict in trajdict_l:
        node_features = trajdict["traj"].float()
        id_features = trajdict["point_ids"]
        adj = trajdict["adj"]
        flag = trajdict["flag"]

        # Create the DGL Graph
        g = dgl.from_scipy(adj, eweight_name="feat")

        padding_node_num = g.num_nodes() - node_features.shape[0]
        padding_node = torch.zeros((padding_node_num, node_features.shape[1])).float()
        all_node_features = torch.cat((node_features, padding_node), dim=0)

        g.ndata["feat"] = all_node_features
        g.ndata["flag"] = flag
        g.ndata["id"] = id_features
        
        g = laplacian_positional_encoding_dense(g, adj, d_lap_pos)
        
        # rtn.append(copy.deepcopy(g)) # yc: this is previous implementation
        rtn.append(g)

    return rtn

