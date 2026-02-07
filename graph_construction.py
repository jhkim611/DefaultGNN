import os
import pandas as pd
import numpy as np
import argparse
import torch
import time
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2021)
args = parser.parse_args()

year = args.year
start_total = time.time()
print(f"begin graph construction for year {year}")

df = pd.read_parquet(f'./data/filtered4/budo_{year}.parquet', engine='fastparquet')
labels = pd.read_parquet(f'./data/labels/budo_{year}.parquet', engine='fastparquet')
comps = set(labels['company'].unique())

df['B'] = np.where(df['TY_MTH']==1, df['O_NO_BIZ'], df['NO_BISOCIAL'])
df['A'] = np.where(df['TY_MTH']==1, df['NO_BISOCIAL'], df['O_NO_BIZ'])
df = df[['A', 'B', 'YYYYMM', 'T_MON_MN_MNAM']]

df['indicator'] = pd.factorize(list(zip(df['A'], df['YYYYMM'])))[0]
df['indicator2'] = pd.factorize(list(zip(df['B'], df['YYYYMM'])))[0]
sum_df = df.groupby('indicator')['T_MON_MN_MNAM'].sum().reset_index()
sum_df2 = df.groupby('indicator2')['T_MON_MN_MNAM'].sum().reset_index()
sum_df = sum_df.rename(columns={'T_MON_MN_MNAM': 'SUM_MONTH'})
sum_df2 = sum_df2.rename(columns={'T_MON_MN_MNAM': 'SUM_MONTH2'})
newdf = pd.merge(df, sum_df, on='indicator', how='left')
newdf = pd.merge(newdf, sum_df2, on='indicator2', how='left')
newdf['ratio'] = newdf['T_MON_MN_MNAM'] / newdf['SUM_MONTH']
newdf['ratio2'] = newdf['T_MON_MN_MNAM'] / newdf['SUM_MONTH2']
newdf = newdf[['A', 'B', 'ratio', 'ratio2']]
df = newdf[newdf['A'].isin(comps) & newdf['B'].isin(comps)]

company_to_index = labels.reset_index().set_index('company')['index']
index_to_company = {idx: val for val, idx in company_to_index.items()}

df['A'] = df['A'].map(company_to_index)
df['B'] = df['B'].map(company_to_index)

edge_index = torch.tensor([df['A'].values, df['B'].values], dtype=torch.long)
edge_weight = torch.tensor(df['ratio'].values, dtype=torch.float)

edge_index2 = torch.tensor([df['B'].values, df['A'].values], dtype=torch.long)
edge_weight2 = torch.tensor(df['ratio2'].values, dtype=torch.float)

print(f'    Edge count: {edge_index.shape[1]}, {edge_index2.shape[1]}')

x = torch.load(f'./data/features/budo_{year}_selected.pt')

if 'future_budo' in labels.columns:
    y = torch.tensor(labels['future_budo'].values, dtype=torch.long)
else:
    y = None

y_unclr = torch.tensor(labels['past_budo_unclr'].values, dtype=torch.long)

print(f'    Node count: {len(labels)}')

num_nodes = len(y)
y_np = y.numpy()

class_0_idx = np.where(y_np == 0)[0]
class_1_idx = np.where(y_np == 1)[0]

np.random.seed(42)
np.random.shuffle(class_0_idx)
np.random.shuffle(class_1_idx)

def split_indices(indices, ratios=(0.7, 0.1, 0.2)):
    n = len(indices)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]

train_0, val_0, test_0 = split_indices(class_0_idx)
train_1, val_1, test_1 = split_indices(class_1_idx)

train_idx = np.concatenate([train_0, train_1])
val_idx = np.concatenate([val_0, val_1])
test_idx = np.concatenate([test_0, test_1])

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

folder_path = './data/graphs'
os.makedirs(folder_path, exist_ok=True)
filename = f'budo_{year}.pt'
filepath = os.path.join(folder_path, filename)
torch.save({
    'edge_index': edge_index,
    'edge_index2': edge_index2,
    'edge_weight': edge_weight,
    'edge_weight2': edge_weight2,
    'x': x,
    'y': y,
    'y_unclr': y_unclr,
    'train_mask': train_mask,
    'val_mask': val_mask,
    'test_mask': test_mask,
    'index_to_company': index_to_company
}, filepath)

elapsed = time.time() - start_total
if elapsed >= 60:
    print(f"graph construction done, elapsed time: {int(elapsed//60)}m {elapsed%60:.2f}s\n")
else:
    print(f"graph construction done, elapsed time: {elapsed:.2f}s\n")