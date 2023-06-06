import argparse
from input_data import load_data,load_labels
from trainGcn import train_gcn
from trainNN import train_nn
from evaluation import get_results
import numpy as np
import pandas as pd
import os
import torch
from preprocessing import PFPDataset,collate
import json
from torch.utils.data import DataLoader

def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))

def train(args):
    # load feature dataframe
    print("loading features...")
    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species, "features.pkl"))
    device = torch.device('cuda:7')
    #uniprot = pd.read_pickle(os.path.join(args.data_path, args.species, "test.pkl"))
    #uniprot[:100].to_pickle('./data/human/test.pkl')
    embeddings_list = []  # VGAE得到的embedings
    for graph in args.graphs:
        print("#############################")
        print("Training",graph)
        adj, features = load_data(graph, uniprot, args)
        embeddings = train_gcn(features, adj, args, graph,device)
        embeddings_list.append(embeddings.cpu().detach().numpy())
    embeddings = np.hstack(embeddings_list)  # 两个embeddings拼接
    np.random.seed(5959)

    cc, mf, bp = load_labels(uniprot)
    # split data into train and test
    num_test = int(np.floor(cc.shape[0] / 5.))  # floor函数向下取整  #测试集数量
    num_train = cc.shape[0] - num_test          #训练集数量
    all_idx = list(range(cc.shape[0]))
    np.random.shuffle(all_idx)

    train_idx = all_idx[:num_train]          #训练集索引
    test_idx = all_idx[num_train:(num_train + num_test)]#测试集索引

    Y_train_cc = cc[train_idx]  # 12107，728   #用于预测cc功能的训练集标签（真实值）
    Y_train_bp = bp[train_idx]  # 12107，5000+   #用于预测bp功能的训练集标签（真实值）
    Y_train_mf = mf[train_idx]                  #用于预测mf功能的训练集标签（真实值）

    Y_test_cc = cc[test_idx]                  #用于预测cc功能的测试集标签（真实值）
    Y_test_bp = bp[test_idx]                  #用于预测bf功能的测试集标签（真实值）
    Y_test_mf = mf[test_idx]                  #用于预测mf功能的测试集标签（真实值）

    X_train = embeddings[train_idx]  # 12107，800            #训练集的embeddings
    X_test = embeddings[test_idx]  # 3026，800               #测试集的embeddings

    train_data_cc = PFPDataset(train_data_X=X_train,train_data_Y=Y_train_cc)
    train_data_bp = PFPDataset(train_data_X=X_train, train_data_Y=Y_train_bp)
    train_data_mf = PFPDataset(train_data_X=X_train, train_data_Y=Y_train_mf)

    #test_data_cc = PFPDataset(train_data_X=X_test,train_data_Y=Y_train_cc)
    #test_data_bp = PFPDataset(train_data_X=X_test,train_data_Y=Y_test_bp)
    #test_data_mf = PFPDataset(train_data_X=X_test,train_data_Y=Y_test_mf)

    dataset_train_cc = DataLoader(train_data_cc, batch_size=128, shuffle=True, collate_fn=collate, drop_last=False)
    dataset_train_bp = DataLoader(train_data_bp, batch_size=128, shuffle=False, collate_fn=collate, drop_last=False)
    dataset_train_mf = DataLoader(train_data_mf, batch_size=128, shuffle=False,collate_fn=collate, drop_last=False)

    #dataset_test_cc = DataLoader(test_data_cc, batch_size=128, shuffle=False, num_workers=16,
    #                           collate_fn=collate, drop_last=False)
    # dataset_test_bp = DataLoader(test_data_bp, batch_size=128, shuffle=False, num_workers=16,
    #                              collate_fn=collate, drop_last=False)
    # dataset_test_mf = DataLoader(test_data_mf, batch_size=128, shuffle=False, num_workers=16,
    #                              collate_fn=collate, drop_last=False)


    print("Start running supervised model...")
    save_path = os.path.join(args.data_path, args.species,"results_new/results_graph2go_" + args.supervised + "_" + ";".join(args.graphs) + "_" + str(args.ppi_attributes) + "_" + str(args.simi_attributes) + "_" + str(args.thr_combined) + "_" + str(args.thr_evalue))


    if args.supervised == "nn":
        # print(embeddings.shape[1],'#################')
        # print(Y_train_cc.shape[1],'#################')

        print("###################################")
        print('----------------------------------')
        print('CC')
        y_score_cc = train_nn(args=args,device=device,input_dim=embeddings.shape[1],output_dim=Y_train_cc.shape[1],train_loader=dataset_train_cc,X_test=X_test,Y_test=Y_test_cc)

        perf_cc = get_results(cc, Y_test_cc, y_score_cc.cpu().numpy())

        with open(save_path + "_cc.json", "w") as f:
            json.dump(perf_cc, f)

        print("###################################")
        print('----------------------------------')
        print('BP')
        y_score_bp = train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_bp.shape[1],
                              train_loader=dataset_train_bp, X_test=X_test, Y_test=Y_test_bp)

        perf_bp = get_results(bp, Y_test_bp, y_score_bp.cpu().numpy())

        with open(save_path + "_bp.json", "w") as f:
            json.dump(perf_bp, f)

        print("###################################")
        print('----------------------------------')
        print('MF')
        y_score_mf = train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_mf.shape[1],
                              train_loader=dataset_train_mf, X_test=X_test, Y_test=Y_test_mf)

        perf_mf = get_results(mf, Y_test_mf, y_score_mf.cpu().numpy())

        with open(save_path + "_mf.json", "w") as f:
            json.dump(perf_mf, f)


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # global parameters
    parser.add_argument('--ppi_attributes', type=int, default=6, help="types of attributes used by ppi.")
    parser.add_argument('--simi_attributes', type=int, default=5, help="types of attributes used by simi.")
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")],
                        default=['ppi', 'sequence_similarity'], help="lists of graphs to use.")
    parser.add_argument('--species', type=str, default="human", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="./data/", help="path storing data.")
    parser.add_argument('--thr_combined', type=float, default=0.3, help="threshold for combiend ppi network.")
    parser.add_argument('--thr_evalue', type=float, default=1e-4, help="threshold for similarity network.")
    parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
    parser.add_argument('--only_gcn', type=int, default=0, help="0 for training all, 1 for only embeddings.")
    parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")

    # parameters for traing GCN
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--epochs_ppi', type=int, default=200, help="Number of epochs to train ppi.")
    parser.add_argument('--epochs_simi', type=int, default=150, help="Number of epochs to train similarity network.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight for L2 loss on embedding matrix.")
    parser.add_argument('--dropout', type=float, default=0, help="Dropout rate (1 - keep probability).")
    parser.add_argument('--model', type=str, default="gcn_vae", help="Model string.")

    args = parser.parse_args()
    print(args)
    train(args)