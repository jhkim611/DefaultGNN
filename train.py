import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, roc_auc_score
import time
from datetime import datetime
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2021)
parser.add_argument("--model_name", type=str, default="new")
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--ce_weight", type=float, default=0.9)
parser.add_argument("--ew_scaling", type=int, default=1)
parser.add_argument("--alpha_val", type=float, default=50)
parser.add_argument('--lambda_val', type=float, default=0.05)
parser.add_argument("--gpu", type=int, default=1)
args = parser.parse_args()

year = args.year
data_dict = torch.load(f"./final/data/graphs/budo_{year}.pt")

edge_index = data_dict["edge_index"]
edge_weight = data_dict["edge_weight"]
edge_index2 = data_dict["edge_index2"]
edge_weight2 = data_dict["edge_weight2"]
x = data_dict["x"]
y = data_dict["y"]
train_mask = data_dict["train_mask"]
val_mask = data_dict["val_mask"]
test_mask = data_dict["test_mask"]

def log_scale(w, alpha=50, eps=1e-12):
    return torch.log1p(alpha*torch.clamp(w, 0.0, 1.0)) / torch.log1p(torch.tensor(alpha + eps, device=w.device))

if args.ew_scaling==0:
    ew_option = "no_scaling"
elif args.ew_scaling==1:
    ew_option = "log scaling"
    edge_weight = log_scale(edge_weight, alpha=args.alpha_val)
    edge_weight2 = log_scale(edge_weight2, alpha=args.alpha_val)

feat_size = x.shape[1]

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(device)

class DefaultGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gnn1 = GCNConv(in_channels, hidden_channels)
        self.gnn12 = GCNConv(in_channels, hidden_channels)
        
        self.gnn2 = GCNConv(hidden_channels, hidden_channels)
        self.gnn22 = GCNConv(hidden_channels, hidden_channels)
        
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)

        self.head_sell = torch.nn.Linear(hidden_channels, 2)
        self.head_buy = torch.nn.Linear(hidden_channels, 2)

        self.fuse_gate = torch.nn.Linear(2*hidden_channels, 1)

        self.final_mlp = torch.nn.Linear(2*hidden_channels, out_channels)
    
    def forward(self, x1, x2, ri1, ri2, edge_index, edge_attr, edge_index2, edge_attr2):
        h = self.gnn1(x1, edge_index, edge_attr)
        h = F.relu(h)
        h = self.gnn12(h, edge_index, edge_attr)
        h = self.norm1(h)
        out = F.relu(h)

        out = out[ri1]

        h2 = self.gnn2(x2, edge_index2, edge_attr2)
        h2 = F.relu(h2)
        h2 = self.gnn22(h2, edge_index2, edge_attr2)
        h2 = self.norm2(h2)
        out2= F.relu(h2)

        out2 = out2[ri2]

        out_sell = self.head_sell(out)
        out_buy = self.head_buy(out2)

        g = torch.sigmoid(self.fuse_gate(torch.cat([out, out2], dim=-1)))
        out_concat = torch.cat([g*out, (1-g)*out2], dim=-1)

        fin_out = self.final_mlp(out_concat)

        return fin_out, out_sell, out_buy

hidden_size = args.hidden_size

if args.model_name=="new":
    model = DefaultGNN(feat_size, hidden_size, 2).to(device)
else:
    modelname = args.model_name
    model = DefaultGNN(feat_size, hidden_size, 2).to(device)
    model.load_state_dict(torch.load(f"./results/saved_models/{modelname}.pt"))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.ce_weight==0.5:
    class_weights = None
else:
    class_weights = torch.tensor([1 - args.ce_weight, args.ce_weight]).to(device)

criterion1 = nn.CrossEntropyLoss(weight=class_weights)

x = x.to(device)
y = y.to(device)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)
edge_index2 = edge_index2.to(device)
edge_weight2 = edge_weight2.to(device)

batch_size = args.batch_size

train_nodes = train_mask.nonzero(as_tuple=False).view(-1)
val_nodes = val_mask.nonzero(as_tuple=False).view(-1)
test_nodes = test_mask.nonzero(as_tuple=False).view(-1)

train_loader = DataLoader(train_nodes, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_nodes, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_nodes, batch_size=batch_size, shuffle=False)

def evaluate(loader):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for batch_nodes in loader:
            batch_nodes = batch_nodes.to(device)

            subset_nodes, sub_edge_index, _, edge_mask = k_hop_subgraph(
                node_idx=batch_nodes,
                num_hops=2,
                edge_index=edge_index,
                relabel_nodes=True,
                num_nodes=None
            )
            sub_edge_weight = edge_weight[edge_mask]
            sub_edge_index = sub_edge_index.to(device)
            sub_edge_weight = sub_edge_weight.reshape(-1,1).to(device)

            subset_nodes2, sub_edge_index2, _, edge_mask2 = k_hop_subgraph(
                node_idx=batch_nodes,
                num_hops=2,
                edge_index=edge_index2,
                relabel_nodes=True,
                num_nodes=None
            )
            sub_edge_weight2 = edge_weight2[edge_mask2]
            sub_edge_index2 = sub_edge_index2.to(device)
            sub_edge_weight2 = sub_edge_weight2.reshape(-1,1).to(device)

            x1, x2 = x[subset_nodes], x[subset_nodes2]
            ri1, ri2 = (subset_nodes.unsqueeze(1)==batch_nodes.unsqueeze(0)).int().argmax(dim=0), (subset_nodes2.unsqueeze(1)==batch_nodes.unsqueeze(0)).int().argmax(dim=0)

            optimizer.zero_grad()            
            out, _, _ = model(x1, x2, ri1, ri2, sub_edge_index, sub_edge_weight, sub_edge_index2, sub_edge_weight2)
            prob = F.softmax(out, dim=1)[:, 1].cpu()
            pred = out.argmax(dim=1).cpu()

            ys.append(y[batch_nodes].cpu())
            preds.append(pred)
            probs.append(prob)

    y_true = torch.cat(ys)
    y_pred = torch.cat(preds)
    y_prob = torch.cat(probs)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return acc, auc, y_true, y_pred

if args.model_name=="new":
    print(f"train new model on year {year}")
    filename = f"model_{year}"
else:
    print(f"train {modelname} on year {year}")
    filename = f"{modelname.split('/')[1]}_{year}"

print(f"Model parameters: hidden size - {args.hidden_size}, batch size - {args.batch_size}, learning rate - {args.lr}, bce weight - {args.ce_weight}, edge scaling - {ew_option} (params: {args.alpha_val}), regularization loss weight - {args.lambda_val}")

best_val_metric = -1
patience = 10
counter = 0
best_epoch = 0
save_date = datetime.today().strftime("%y%m%d")
start_time = time.time()

for epoch in range(1, 501):
    model.train()
    total_loss = 0

    for batch_nodes in train_loader:
        batch_nodes = batch_nodes.to(device)

        subset_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=batch_nodes,
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=None
        )
        sub_edge_weight = edge_weight[edge_mask]
        sub_edge_index = sub_edge_index.to(device)
        sub_edge_weight = sub_edge_weight.reshape(-1,1).to(device)

        subset_nodes2, sub_edge_index2, mapping, edge_mask2 = k_hop_subgraph(
            node_idx=batch_nodes,
            num_hops=2,
            edge_index=edge_index2,
            relabel_nodes=True,
            num_nodes=None
        )
        sub_edge_weight2 = edge_weight2[edge_mask2]
        sub_edge_index2 = sub_edge_index2.to(device)
        sub_edge_weight2 = sub_edge_weight2.reshape(-1,1).to(device)

        x1, x2 = x[subset_nodes], x[subset_nodes2]
        ri1, ri2 = (subset_nodes.unsqueeze(1)==batch_nodes.unsqueeze(0)).int().argmax(dim=0), (subset_nodes2.unsqueeze(1)==batch_nodes.unsqueeze(0)).int().argmax(dim=0)

        optimizer.zero_grad()            
        out, out_s, out_b = model(x1, x2, ri1, ri2, sub_edge_index, sub_edge_weight, sub_edge_index2, sub_edge_weight2)
        loss1 = criterion1(out, y[batch_nodes])

        p_s = F.softmax(out_s, dim=1)[:, 1]
        p_b = F.softmax(out_s, dim=1)[:, 1]

        conf_s = torch.abs(p_s - 0.5)*2.0
        conf_b = torch.abs(p_b - 0.5)*2.0

        conf_w = conf_s*conf_b
        diff = (p_s - p_b).pow(2)

        loss2 = (conf_w*diff).sum()/(conf_w.sum() + 1e-8)

        loss = loss1 + args.lambda_val*loss2

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_nodes.size(0)

    train_loss = total_loss / train_nodes.size(0)

    val_acc, val_metric, _, _ = evaluate(val_loader)

    if epoch % 10 == 0:
        elapsed = time.time() - start_time
        if elapsed >= 60:
            print(f"Epoch {epoch:03d} - Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f} | Val AUC: {val_metric:.4f} | Val AR: {2*val_metric - 1:.4f} | elapsed time: {int(elapsed//60)}m {elapsed%60:.2f}s")
        else:
            print(f"Epoch {epoch:03d} - Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f} | Val AUC: {val_metric:.4f} | Val AR: {2*val_metric - 1:.4f} | elapsed time: {elapsed:.2f}s")

    if val_metric > best_val_metric:
        best_val_metric = val_metric
        counter = 0
        best_epoch = epoch

        save_dir = f"./results/saved_models/{save_date}"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename+".pt")
        torch.save(model.state_dict(), filepath)
        print(f"best model saved at epoch {epoch:03d}")
    else:
        counter += 1
        if counter >= patience:
            if best_epoch != epoch - counter:
                save_dir = f"./results/saved_models/{save_date}"
                os.makedirs(save_dir, exist_ok=True)
                filepath = os.path.join(save_dir, filename+".pt")
                torch.save(model.state_dict(), filepath)
            print(f"early stopping at epoch {epoch}")
            break

elapsed = time.time() - start_time
if elapsed >= 60:
    print(f"training done ({save_dir}/{filename}.pt), elapsed time: {int(elapsed//60)}m {elapsed%60:.2f}s\n")
else:
    print(f"training done ({save_dir}/{filename}.pt), elapsed time: {elapsed:.2f}s\n")