import pandas as pd
from pyvis.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import GCNConv
from datetime import datetime
import time
import os
import argparse
import warnings
import networkx as nx
from captum.attr import IntegratedGradients
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
parser.add_argument('--pred_file', type=str, default='none')
parser.add_argument('--target', type=int, default=1010108767)
args = parser.parse_args()

year = args.year
data_dict = torch.load(f'./data/graphs/budo_{year}.pt')

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

x = x.to(device)
y = y.to(device)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)
edge_index2 = edge_index2.to(device)
edge_weight2 = edge_weight2.to(device)

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
model = DefaultGNN(feat_size, hidden_size, 2, heads=args.heads, beta=args.beta, bias=bool(args.bias)).to(device)
model.load_state_dict(torch.load(f"./saved_models/{modelname}.pt"))
model.eval()


def visualize_pg_explanation_pyvis(model, prediction_data, target, x, y, edge_index, edge_weight, edge_index2, edge_weight2, num_gnn_layers, device, K=10):
    node_to_explain = prediction_data.loc[prediction_data['company']==target]['node index'].values[0].item()
    
    subset, sub_edge_index, _, _ = k_hop_subgraph(
                        node_idx=node_to_explain,
                        num_hops=num_gnn_layers,
                        edge_index=edge_index,
                        relabel_nodes=True,
                        num_nodes=None
                    )
    subset2, sub_edge_index2, _, _ = k_hop_subgraph(
        node_idx=node_to_explain, num_hops=num_gnn_layers, 
        edge_index=edge_index2, relabel_nodes=True, num_nodes=None
    )

    subset = torch.unique(torch.cat([subset, subset2]))
    sub_edge_index = torch.cat([sub_edge_index, sub_edge_index2], dim=1)
    sub_edge_index = torch.unique(sub_edge_index, dim=1)

    local_node_idx = (subset==node_to_explain).nonzero().item()
    sub_x = x[subset].to(device)
    sub_edge_index = sub_edge_index.to(device)
    sub_target = y[subset].to(device)

    explanation_edge_mask = torch.ones(sub_edge_index.size(1))
    sub_edge_index_cpu = sub_edge_index.cpu()
    sub_target_cpu = sub_target.cpu()
    sub_x_cpu = sub_x.cpu()

    G = nx.DiGraph()
    num_nodes_in_subgraph = sub_x_cpu.size(0)
    G.add_nodes_from(range(num_nodes_in_subgraph))
    edge_list_cpu = sub_edge_index_cpu.t().tolist()
    edge_mask_list_cpu = explanation_edge_mask.tolist()

    for i in range(len(edge_list_cpu)):
        u, v = edge_list_cpu[i]
        mask_value = edge_mask_list_cpu[i]
        if u != v:
            G.add_edge(u, v, mask=mask_value)

    start_1 = time.time()
    top_k_edges_tuples, global_to_local = explain_target_node_tuples(
            model=model,
            target_node_idx=node_to_explain,
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_index2=edge_index2,
            edge_weight2=edge_weight2,
            y=y,
            num_hops=num_gnn_layers,
            top_k=K,
            threshold=0
        )

    local_to_global = {v: k for k, v in global_to_local.items()}
   
    top_k_edges_set = set(top_k_edges_tuples)

    top_k_node_set = set()
    for u, v in top_k_edges_tuples:
        top_k_node_set.add(u)
        top_k_node_set.add(v)
    top_k_node_set.add(local_node_idx)

    start = time.time()
    prediction = prediction_data.loc[prediction_data['company']==target]['future_budo_prob'].values[0].item()
    nt = Network(notebook=True, directed=True, height="750px", width="100%")

    class_colors = ['#2ECC71', '#E74C3C']
    top_k_edge_rgb = (52,73,94)

    top_k_label = []
    for i in G.nodes():
        global_node_idx = local_to_global[i]

        class_label = 0
        if i == local_node_idx:
            class_label = sub_target_cpu[i].item()
            title_text = f"TARGET Node {node_to_explain}"
            border_color= '#FF0000'
            node_shape='star'
        else:
            try:
                class_label = prediction_data.loc[global_node_idx, 'past_budo']
            except KeyError:
                class_label = 0
            title_text = f"Node {global_node_idx}"
            border_color = '#AAAAAA'
            node_shape='dot'

        node_color = class_colors[int(class_label)%2]
        is_highlighted = i in top_k_node_set

        if i == local_node_idx:
            node_size = 20
        elif is_highlighted:
            node_size = 15
            top_k_label.append(class_label)
        else:
            node_size = 8

        node_label_font_style = {
            'color':'#000000',
            'background': '#FFFFFF',
            'strokeWidth': 4,
            'strokeColor': '#FFFFFF',
            'size': 12
        }

        nt.add_node(
            n_id=i,
            size=node_size,
            shape=node_shape,
            color={
                'border': node_color,
                'background': node_color,
                'highlight': node_color
            },
            borderWidth= 1,
            title=title_text,
            opacity=1.0 if is_highlighted else 0.2,
            font=node_label_font_style
        )
    
    for u, v, data in G.edges(data=True):
        is_top_k = (u, v) in top_k_edges_set
        mask_val = data['mask']

        if is_top_k:
            edge_width = 8.0
            edge_color = f'rgba({top_k_edge_rgb[0]}, {top_k_edge_rgb[1]}, {top_k_edge_rgb[2]}, 1.0)'
            title_text = f"Mask Value: {mask_val:.4f}"

            nt.add_edge(
            source=u,
            to=v,
            width=edge_width,
            color=edge_color,
            title=title_text
            )
        else:
            edge_width = 1.0
            edge_color = 'rgba(189, 195, 199, 0.4)'
            title_text = f"Mask Value: {mask_val:.4f}"

            nt.add_edge(
                source=u,
                to=v,
                width=edge_width,
                color=edge_color,
                title=title_text,
                arrows=None
            )
        
    nt.set_options("""
    var options = {
                   "physics": {
                   "enabled": true,
                   "barnesHut": {
                   "gravitationalConstant": -8000,
                   "centralGravity": 0.3,
                   "springLength": 95,
                   "springConstant": 0.04,
                   "damping": 0.09,
                   "avoidOverlap": 0.1
                   },
                   "minVelocity": 0.75
                   }
                   }               
    """)

    save_date = datetime.today().strftime("%y%m%d")
    output_dir = f'./results/figures'
    filename = f"visualization_{target}.html"
    full_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    nt.save_graph(full_path)

    elapsed = time.time() - start
    if elapsed >= 60:
        print(f"visualization done, elapsed time: {int(elapsed//60)}m {elapsed%60:.2f}s")
    else:
        print(f"visualization done, elapsed time: {elapsed:.2f}s")

class MultiplexIGWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def model_forward(self, edge_mask1, edge_mask2, 
                      sub_x1, sub_x2, 
                      target_ri1, target_ri2, 
                      sub_edge_index1, sub_edge_attr1, 
                      sub_edge_index2, sub_edge_attr2):
        
        e_mask1 = edge_mask1.squeeze(0) if edge_mask1.dim() > 1 else edge_mask1
        e_mask2 = edge_mask2.squeeze(0) if edge_mask2.dim() > 1 else edge_mask2

        weighted_attr1 = sub_edge_attr1 * e_mask1.unsqueeze(-1)
        weighted_attr2 = sub_edge_attr2 * e_mask2.unsqueeze(-1)

        return self.model(
            x1=sub_x1, x2=sub_x2, 
            ri1=target_ri1, ri2=target_ri2, 
            edge_index=sub_edge_index1, 
            edge_attr=weighted_attr1,
            edge_index2=sub_edge_index2, 
            edge_attr2=weighted_attr2
        )

def explain_target_node_tuples(model, target_node_idx, x, 
                               edge_index, edge_weight, 
                               edge_index2, edge_weight2, 
                               y=None, num_hops=2, top_k=10, threshold=0.0001):
    wrapper = MultiplexIGWrapper(model)
    device = x.device

    subset1, sub_edge_index1, mapping1, edge_mask1 = k_hop_subgraph(
        node_idx=target_node_idx, num_hops=num_hops, 
        edge_index=edge_index, relabel_nodes=True
    )
    sub_weight1 = edge_weight[edge_mask1].view(-1, 1) 

    subset2, sub_edge_index2, mapping2, edge_mask2 = k_hop_subgraph(
        node_idx=target_node_idx, num_hops=num_hops, 
        edge_index=edge_index2, relabel_nodes=True
    )
    sub_weight2 = edge_weight2[edge_mask2].view(-1, 1)

    combined_subset = torch.unique(torch.cat([subset1, subset2]))
    
    other_nodes = combined_subset[combined_subset != target_node_idx].sort()[0]
    
    new_ordered_nodes = torch.cat([torch.tensor([target_node_idx], device=device), other_nodes])

    global_to_local = {int(node): i for i, node in enumerate(new_ordered_nodes)}

    ig = IntegratedGradients(wrapper.model_forward)
    
    if y is not None:
        target_label = y[target_node_idx].item()
    else:
        with torch.no_grad():
            out = model(x[subset1], x[subset2], mapping1, mapping2, 
                        sub_edge_index1, sub_weight1, 
                        sub_edge_index2, sub_weight2)
            target_label = out.argmax(dim=1).item()

    attr1, attr2 = ig.attribute(
        inputs=(torch.ones(sub_edge_index1.size(1), device=device).unsqueeze(0),
                torch.ones(sub_edge_index2.size(1), device=device).unsqueeze(0)),
        baselines=(torch.zeros(sub_edge_index1.size(1), device=device).unsqueeze(0),
                   torch.zeros(sub_edge_index2.size(1), device=device).unsqueeze(0)),
        target=target_label,
        additional_forward_args=(x[subset1], x[subset2], mapping1, mapping2, 
                                 sub_edge_index1, sub_weight1, 
                                 sub_edge_index2, sub_weight2),
        n_steps=200,
        internal_batch_size=1 
    )

    attr1 = attr1.squeeze(0)
    attr2 = attr2.squeeze(0)

    all_candidate_edges = []

    scores1 = attr1.abs().cpu().detach().numpy()
    e_idx1 = sub_edge_index1.cpu().numpy()
    nodes1 = subset1.cpu().numpy()
    for i in range(len(scores1)):
        u_global, v_global = int(nodes1[e_idx1[0, i]]), int(nodes1[e_idx1[1, i]])
        all_candidate_edges.append((global_to_local[u_global], global_to_local[v_global], scores1[i]))

    scores2 = attr2.abs().cpu().detach().numpy()
    e_idx2 = sub_edge_index2.cpu().numpy()
    nodes2 = subset2.cpu().numpy()
    for i in range(len(scores2)):
        u_global, v_global = int(nodes2[e_idx2[0, i]]), int(nodes2[e_idx2[1, i]])
        all_candidate_edges.append((global_to_local[u_global], global_to_local[v_global], scores2[i]))

    all_candidate_edges.sort(key=lambda x: x[2], reverse=True)

    top_k_edges_tuples = []
    for u_local, v_local, score in all_candidate_edges:
        if len(top_k_edges_tuples) >= top_k or score < threshold:
            break
        top_k_edges_tuples.append((u_local, v_local))

    return top_k_edges_tuples, global_to_local
    
pred_df = pd.read_parquet(f'./pred_results/{args.pred_file}.parquet', engine='fastparquet')

visualize_pg_explanation_pyvis(model=model, prediction_data=pred_df, target=args.target, x=x, y=y, edge_index=edge_index, edge_weight=edge_weight, edge_index2=edge_index2, edge_weight2=edge_weight2, num_gnn_layers=2, device=device)