import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

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
index_to_company = data_dict["index_to_company"]

def log_scale(w, alpha=50, eps=1e-12):
    return torch.log1p(alpha*torch.clamp(w, 0.0, 1.0)) / torch.log1p(torch.tensor(alpha + eps, device=w.device))

if args.ew_scaling==0:
    ew_option = "no_scaling"
elif args.ew_scaling==1:
    ew_option = "log scaling"
    edge_weight = log_scale(edge_weight, alpha=args.alpha_val)
    edge_weight2 = log_scale(edge_weight2, alpha=args.alpha_val)

feat_size = x.shape[1]

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
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

modelname = args.model_name
model = DefaultGNN(feat_size, hidden_size, 2).to(device)
model.load_state_dict(torch.load(f"./results/saved_models/{modelname}.pt"))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.ce_weight==0.5:
    class_weights = None
else:
    class_weights = torch.tensor([1 - args.ce_weight, args.ce_weight]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

x = x.to(device)
if y!=None:
    y = y.to(device)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)
edge_index2 = edge_index2.to(device)
edge_weight2 = edge_weight2.to(device)

batch_size = args.batch_size

if args.predict_for=='test':
    test_nodes = test_mask.nonzero(as_tuple=False).view(-1)
    test_loader = DataLoader(test_nodes, batch_size=batch_size, shuffle=False)

    y2 = data_dict['y_unclr']
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    unclr_mask = (y2[test_idx]==0)

elif args.predict_for=='full':
    test_loader = DataLoader(torch.arange(x.shape[0]), batch_size=batch_size, shuffle=False)
    y2 = data_dict['y_unclr']
    unclr_mask = (y2==0)

def evaluate(loader):
    model.eval()
    ys, probs, comps, inds = [], [], [], []
    with torch.no_grad():
        for batch_nodes in loader:
            batch_nodes = batch_nodes.to(device)
            comp = torch.tensor([index_to_company[int(c)] for c in batch_nodes])

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

            if y!=None:
                ys.append(y[batch_nodes].cpu())
            probs.append(prob)
            comps.append(comp)
            inds.append(batch_nodes.cpu())

    if y!=None:
        y_true = torch.cat(ys)
    else:
        y_true = None
    y_prob = torch.cat(probs)
    companies = torch.cat(comps)
    indices = torch.cat(inds)

    return y_true, y_prob, companies, indices

print(f'predicting form year {year} with model {modelname}.pt')

y_true, y_prob, companies, indices = evaluate(test_loader)

if y!=None:
    df = pd.DataFrame({'company': companies.tolist(), 'node index': indices.tolist(), 'unclr_budo': [0 if m else 1 for m in unclr_mask.tolist()], 'future_budo_prob': y_prob.tolist(), 'label': y_true.tolist()})
else:
    df = pd.DataFrame({'company': companies.tolist(), 'node index': indices.tolist(), 'unclr_budo': [0 if m else 1 for m in unclr_mask.tolist()], 'future_budo_prob': y_prob.tolist()})

save_date = datetime.today().strftime("%y%m%d")
save_dir = f"./results/saved_preds/{save_date}"
os.makedirs(save_dir, exist_ok=True)
filename = f"budo_{year}_{args.predict_for}_trainedon_{modelname.replace('/', '_')}.parquet"
filepath = os.path.join(save_dir, filename)
df.to_parquet(filepath)

result_dir = f'./results/performances/{save_date}'
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir, f'budo_{year}.txt')

def compute_metrics(df_subset, name, threshold=0.5):
    if len(df_subset)==0:
        print(f"{name}: no rows")
        return

    labels = df_subset["label"].values
    probs = df_subset["future_budo_prob"].values
    preds = (probs>=threshold).astype(int)

    acc = accuracy_score(labels, preds)

    if len(set(labels))>1:
        auc = roc_auc_score(labels, probs)
    else:
        auc = float("nan")

    print(f"{name} - Count: {len(df_subset):,} | Accuracy: {acc:.4f} | AUC: {auc:.4f} | AR: {2*auc-1:.4f}")
    with open(result_file, 'a') as f:
        f.write(f"{name} - Count: {len(df_subset):,} | Accuracy: {acc:.4f} | AUC: {auc:.4f} | AR: {2*auc-1:.4f}\n")

if y!=None:
    with open(result_file, 'a') as f:
        f.write(f"model name = {modelname.replace('/', '_')}\n")

    compute_metrics(df, "All")

    compute_metrics(df[df["unclr_budo"] == 0], "NoHist")

    with open(result_file, 'a') as f:
        f.write(f"__________________________________\n")

print("prediction done")