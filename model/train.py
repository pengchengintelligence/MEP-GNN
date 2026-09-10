from model import *
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import csv
import random
from util import *
import matplotlib
import os
import datetime
import argparse

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('--adj_Sample', type=float, default=0.1)
parser.add_argument('--batch_size_', type=int, default=30)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--x', type=int, default=1)
parser.add_argument('--data_dir', type=str, default=os.environ.get('MEP_GNN_DATA_DIR', '/model/ROSMAP/data'))
parser.add_argument('--output_dir', type=str, default=os.environ.get('MEP_GNN_OUTPUT_DIR', '/model/ROSMAP/result/VCP'))
parser.add_argument('--index_dir', type=str, default='', help='Subdirectory under data_dir that stores *_all_index.txt files.')
parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

device = torch.device(args.device)
if device.type == 'cuda':
    torch.cuda.set_device(device)

print("Using device:", device)
if device.type == 'cuda':
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(device))

index_dir_name = args.index_dir

base_dir = args.output_dir
save_dir = os.path.join(base_dir, index_dir_name)
os.makedirs(save_dir, exist_ok=True)

adj_1 = os.path.join(args.data_dir, '1_tr_adj.csv')
adj_2 = os.path.join(args.data_dir, '2_tr_adj.csv')
adj_3 = os.path.join(args.data_dir, '3_tr_adj.csv')
    
adj_matrix1 = pd.read_csv(adj_1, index_col=0).values.astype(float)
adj_1 = torch.LongTensor((adj_matrix1 > args.adj_Sample).astype(int)).to(device)
adj_matrix2 = pd.read_csv(adj_2, index_col=0).values.astype(float)
adj_2 = torch.LongTensor((adj_matrix2 > args.adj_Sample).astype(int)).to(device)
adj_matrix3 = pd.read_csv(adj_3, index_col=0).values.astype(float)
adj_3 = torch.LongTensor((adj_matrix3 > args.adj_Sample).astype(int)).to(device)
    
tr_path = [
    os.path.join(args.data_dir, '1_tr.csv'),
    os.path.join(args.data_dir, '2_tr.csv'),
    os.path.join(args.data_dir, '3_tr.csv')]
    
te_path = [
    os.path.join(args.data_dir, '1_te.csv'),
    os.path.join(args.data_dir, '2_te.csv'),
    os.path.join(args.data_dir, '3_te.csv')]
    
tr_label_path = os.path.join(args.data_dir, 'labels_tr.csv')
te_label_path = os.path.join(args.data_dir, 'labels_te.csv')
 
tr_dataset = [CustomGeneDataset(path, tr_label_path) for path in tr_path]
te_dataset = [CustomGeneDataset(path, te_label_path) for path in te_path]
    
tr_data = MultiModalWrapperDataset(tr_dataset)
te_data = MultiModalWrapperDataset(te_dataset)
    
batch_size_ = args.batch_size_
tr_data_loader = DataLoader(dataset=tr_data, batch_size=batch_size_, shuffle=True)
te_data_loader = DataLoader(dataset=te_data, batch_size=batch_size_, shuffle=False)
    
index_base_path = os.path.join(args.data_dir, index_dir_name)
index_1 = os.path.join(index_base_path, '1_all_index.txt')
with open(index_1, 'r') as f:
        selected_genes1 = [line.strip() for line in f if line.strip()]
index_2 = os.path.join(index_base_path, '2_all_index.txt')
with open(index_2, 'r') as f:
        selected_genes2 = [line.strip() for line in f if line.strip()]
index_3 = os.path.join(index_base_path, '3_all_index.txt')
with open(index_3, 'r') as f:
        selected_genes3 = [line.strip() for line in f if line.strip()]

all_genes1 = pd.read_csv(tr_path[0], index_col=0).columns.tolist()
all_genes2 = pd.read_csv(tr_path[1], index_col=0).columns.tolist()
all_genes3 = pd.read_csv(tr_path[2], index_col=0).columns.tolist()

topk_indices1 = np.array([all_genes1.index(gene) for gene in selected_genes1 if gene in all_genes1])
topk_indices2 = np.array([all_genes2.index(gene) for gene in selected_genes2 if gene in all_genes2])
topk_indices3 = np.array([all_genes3.index(gene) for gene in selected_genes3 if gene in all_genes3])

topk_indices = [topk_indices1, topk_indices2, topk_indices3]

for i in range(50):    
    num_epochs = 2000
    learning_rate = 1e-3
    weight_decay = 1e-4

    loss_function = nn.CrossEntropyLoss()
    
    input_in_dim = [pd.read_csv(p, index_col=0).shape[1] for p in tr_path]
    
    input_hidden_dim = [64, 64, 64]
    num_class_list = [2, 3, 4]
    network = MEPGNN(num_class=2, num_views=3, hidden_dim=input_hidden_dim, dropout=args.dropout, alpha=args.alpha, in_dim=input_in_dim, topk_indices=topk_indices)
    network.to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)

    best_acc = 0.0
    best_epoch = 0
    best_te_f1 = 0.0
    best_te_auc = 0.0
    best_te_sen = 0.0
    best_te_spe = 0.0

    train_loss_all, train_acc_all, test_acc_all = [], [], []
    train_auc_all, test_auc_all = [], []
    train_f1_all, test_f1_all = [], []
    test_sen_all, test_spe_all = [], []
    best_model_state = None



    for epoch in range(0, num_epochs):
        print(' Epoch {}/{}'.format(epoch, num_epochs - 1))
        print("-" * 10)
        network.train()
        current_loss = 0.0
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        tr_probs_all = []
        tr_labels_all = []
        tr_preds_all = []

        for i, (modal_inputs, targets) in enumerate(tr_data_loader):

            batch_x1 = modal_inputs[0].unsqueeze(-1).to(torch.float32).to(device)
            batch_x2 = modal_inputs[1].unsqueeze(-1).to(torch.float32).to(device)
            batch_x3 = modal_inputs[2].unsqueeze(-1).to(torch.float32).to(device)
            
            targets = targets.long().to(device)

            optimizer.zero_grad()
            loss_fusion, tr_logits, gat_output1, gat_output2, gat_output3, output1, output2, output3 = network(
                batch_x1, batch_x2, batch_x3, adj_1, adj_2, adj_3, targets, epoch=epoch, save_dir=save_dir)

            tr_prob = F.softmax(tr_logits, dim=1)
            tr_pre_lab = torch.argmax(tr_prob, 1)

            loss = loss_fusion
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x1.size(0)
            train_corrects += torch.sum(tr_pre_lab == targets.data)
            train_num += batch_x1.size(0)

            tr_probs_all.extend(tr_prob[:, 1].detach().cpu().numpy()) 
            tr_labels_all.extend(targets.detach().cpu().numpy())
            tr_preds_all.extend(tr_pre_lab.detach().cpu().numpy())

        torch.cuda.empty_cache()

        network.eval()
        test_loss = 0.0
        test_corrects = 0
        test_num = 0
        te_probs_all = []
        te_labels_all = []
        te_preds_all = []

        for i, (modal_inputs, targets) in enumerate(te_data_loader):

            batch_x1 = modal_inputs[0].unsqueeze(-1).to(torch.float32).to(device)
            batch_x2 = modal_inputs[1].unsqueeze(-1).to(torch.float32).to(device)
            batch_x3 = modal_inputs[2].unsqueeze(-1).to(torch.float32).to(device)
            
            targets = targets.long().to(device)

            te_logits = network.infer(batch_x1, batch_x2, batch_x3, adj_1, adj_2, adj_3)
            
            te_prob = F.softmax(te_logits, dim=1)
            te_pre_lab = torch.argmax(te_prob, 1)

            test_corrects += torch.sum(te_pre_lab == targets.data)
            test_num += batch_x1.size(0)

            with torch.no_grad():
                te_logits = network.infer(batch_x1, batch_x2, batch_x3, adj_1, adj_2, adj_3)
                te_prob = F.softmax(te_logits, dim=1)
                te_preds = torch.argmax(te_prob, 1)

            te_probs_all.extend(te_prob[:, 1].detach().cpu().numpy())  
            te_labels_all.extend(targets.detach().cpu().numpy())
            te_preds_all.extend(te_pre_lab.detach().cpu().numpy())

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        test_acc_all.append(test_corrects.double().item() / test_num)
        tr_auc = roc_auc_score(tr_labels_all, tr_probs_all)
        train_auc_all.append(tr_auc)
        
        try:
            te_auc = roc_auc_score(te_labels_all, te_probs_all)
        except:
            te_auc = 0.0
            
        test_auc_all.append(te_auc)
        tr_f1 = f1_score(tr_labels_all, tr_preds_all)
        train_f1_all.append(tr_f1)
        te_f1 = f1_score(te_labels_all, te_preds_all)
        test_f1_all.append(te_f1)
        test_sen = sensitivity_score(te_labels_all, te_preds_all)
        test_sen_all.append(test_sen)
        test_spe = specificity_score(te_labels_all, te_preds_all)
        test_spe_all.append(test_spe)

        if test_acc_all[-1] > best_acc:
            best_acc = test_acc_all[-1]
            best_epoch = epoch + 1
            best_te_f1 = te_f1
            best_te_auc = te_auc
            best_te_sen = test_sen
            best_te_spe = test_spe
            best_model_state = network.state_dict()
    
    model_save_dir = save_dir
    file_name = os.path.join(model_save_dir, f"{best_acc:.3f}_best_model.pth")
    torch.save(best_model_state, file_name)
    print(f"best model save：{file_name}")
    
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    def append_results_to_csv(filename, best_acc, best_te_f1, best_te_auc, best_te_sen, best_te_spe,adj_Sample,dropout,alpha,batch_size_):
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([run_id, best_epoch, best_acc, best_te_f1, best_te_auc, best_te_sen, best_te_spe,adj_Sample,dropout,alpha,batch_size_])

    csv_file = os.path.join(save_dir, 'result.csv')

    append_results_to_csv(csv_file, best_acc, best_te_f1, best_te_auc, best_te_sen, best_te_spe,args.adj_Sample,args.dropout,args.alpha,args.batch_size_)
