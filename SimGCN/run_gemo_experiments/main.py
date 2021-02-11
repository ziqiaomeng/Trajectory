import argparse
import time
import dgl.init
import torch
import torch.nn.functional as F
import utils_data
import random
import numpy as np
from model_zoo import GCN, GAT, SGC, SAGE, ChebNet, SimGCN, MLP, GIN

EPS = 1e-15


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print("success")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='GAT')
    parser.add_argument('--num_hidden', type=int, default=2)
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--num_heads_layer_one', type=int, default=1)
    parser.add_argument('--num_heads_layer_two', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay_layer_one', type=float, default=5e-4)
    parser.add_argument('--weight_decay_layer_two', type=float, default=5e-4)
    parser.add_argument('--num_epochs_patience', type=int, default=200)
    parser.add_argument('--num_epochs_max', type=int, default=1000)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--dataset_split', type=str)
    parser.add_argument('--learning_rate_decay_patience', type=int, default=50)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8)
    args = parser.parse_args()

    if args.dataset_split == 'jknet':
        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = utils_data.load_data(
            args.dataset, None, 0.6, 0.2)
    else:
        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = utils_data.load_data(
            args.dataset, args.dataset_split)
    acc = []
    for seed in range(args.iter):
        setup_seed(seed * 10)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)

        if args.model == 'GCN':   net = GCN(num_features, args.num_hidden, num_labels)
        if args.model == 'GAT':   net = GAT(num_features, args.num_hidden, num_labels)
        if args.model == 'GIN':   net = GIN(num_features, args.num_hidden, num_labels)
        if args.model == 'SGC':   net = SGC(num_features, args.num_hidden, num_labels)
        if args.model == 'SAGE':  net = SAGE(num_features, args.num_hidden, num_labels)
        if args.model == 'ChebNet':   net = ChebNet(num_features, args.num_hidden, num_labels)
        if args.model == 'MLP':   net = MLP(num_features, args.num_hidden, num_labels)
        if args.model == 'SimGCN':    net = SimGCN(num_features, args.num_hidden, num_labels)
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.learning_rate)
        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                             factor=args.learning_rate_decay_factor,
                                                                             patience=args.learning_rate_decay_patience)

        net.cuda()
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

        # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
        patience = args.num_epochs_patience
        vlss_mn = np.inf
        vacc_mx = 0.0
        vacc_early_model = None
        vlss_early_model = None
        state_dict_early_model = None
        curr_step = 0

        # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
        dur = []
        for epoch in range(args.num_epochs_max):
            t0 = time.time()

            net.train()
            # features = torch.load('embeddiings.pt')
            train_logp = net(g, features)
            train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])

            train_pred = train_logp.argmax(dim=1)
            train_acc = torch.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            net.eval()
            with torch.no_grad():
                val_logp = net(g, features)
                val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
                val_pred = val_logp.argmax(dim=1)
                val_acc = torch.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()

            learning_rate_scheduler.step(val_loss)

            dur.append(time.time() - t0)
            if (epoch + 1) % 50 == 0:
                print(
                    "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                        epoch + 1, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))

            # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
            if val_acc >= vacc_mx:
                vacc_early_model = val_acc
                state_dict_early_model = net.state_dict()
                vacc_mx = np.max((val_acc, vacc_mx))
                curr_step = 0
                # torch.save(xmat, 'embeddiings.pt')
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
            state_dict_early_model = net.state_dict()
            if epoch >= 5000:
                break

        net.load_state_dict(state_dict_early_model)
        net.eval()
        with torch.no_grad():
            test_logp = net(g, features)
            # test_logp = F.log_softmax(test_logits, 1)
            test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
            test_pred = test_logp.argmax(dim=1)
            test_acc = torch.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()
        print('>>{}, {}: {:.2f}'.format(args.dataset, args.model, test_acc * 100))
        acc.append(test_acc)
    print('==' * 20)
    print('{:.2f}+{:.2f}'.format(np.mean(acc) * 100, np.std(acc) * 100))
    print('==' * 20)
