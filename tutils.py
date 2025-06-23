import torch 
import numpy as np
import torch.nn as nn
from utils import *
from concept_label_creation import *
from txlstm_szpool import *



def load_model(root, name, fold, subdir, device):
    model_root = os.path.join(root, 'Models', f'split{fold}', subdir, name)
    model = transformer_lstm(device=device)
    trained = os.path.join(model_root, 'model.pth.tar')
    model.load_state_dict(torch.load(trained, map_location=device))
    return model, model_root

def load_model_1nn(root, name, fold, subdir, device):
    model_root = os.path.join(root, 'Models', f'split{fold}', subdir, name)
    model = deepsoz_1nn(device=device)
    trained = os.path.join(model_root, 'model.pth.tar')
    model.load_state_dict(torch.load(trained, map_location=device))
    return model, model_root

def model_test(device, model, model_root, dataloader):
    sig = nn.Sigmoid()
    label_file = pd.read_csv(os.path.join(model_root, 'concepts.csv'))
    label_obj = label_generator()
    true, pred = [], []
    sz_embd, nsz_embd = [], []
    for batch_idx, data in enumerate(dataloader):
        inputs = data['buffers']
        det_labels = data['sz_labels'].long().to(device)
        pt_id = data['patient numbers'][0].item()
        concept_labels = label_obj(label_file, pt_id)
        concept_labels = torch.DoubleTensor(concept_labels).to(device).reshape(1, -1).repeat(inputs.shape[1], 1)
        inputs = inputs.to(device)
        hg, concept_pred, sz_pred, _  = model(inputs)
        B, Nsz, T, L = hg.size()
        h_flatten = hg.view(B * Nsz * T, L)
        h_flatten = F.normalize(h_flatten, dim=1)

        det_labels = det_labels.view(-1)
        # Find indices for seizure and non-seizure cases
        sz_idx = torch.where(det_labels == 1)[0]
        nsz_idx = torch.where(det_labels == 0)[0]

        # Extract embeddings for seizure and non-seizure cases
        h_sz = h_flatten[sz_idx, :] if len(sz_idx) > 0 else torch.empty(0, h_flatten.size(-1), device=h_flatten.device)
        h_sz = h_sz.view(sz_idx.shape[0], L)
        sz_embd.append(h_sz.detach().cpu().numpy())

        h_nsz = h_flatten[nsz_idx, :] if len(nsz_idx) > 0 else torch.empty(0, h_flatten.size(-1), device=h_flatten.device)
        h_nsz = h_nsz.view(nsz_idx.shape[0], L)
        nsz_embd.append(h_nsz.detach().cpu().numpy())

        true.append(concept_labels.detach().cpu().numpy())
        pred.append(sig(concept_pred).detach().cpu().numpy())
    return true, pred, sz_embd, nsz_embd

def confusion_metrics(true, pred):
    N = true.shape[0]
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(N):
        tp += int(true[i] == 1 and pred[i] == 1)
        tn += int(true[i] == 0 and pred[i] == 0)
        fp += int(true[i] == 0 and pred[i] == 1)
        fn += int(true[i] == 1 and pred[i] == 0)

    return tn, fp, fn, tp

def concept_metric(true, pred):
    cat_true = np.concatenate(true)
    cat_pred = np.concatenate(pred)

    accuracy, recall, specificity, f1score, precision = [], [], [], [], []
    N = cat_true.shape[0]

    for i in range(20):
        concept_label = cat_true[:, i]
        pred_label = (cat_pred[:, i] > 0.5).astype(int)
        tn, fp, fn, tp  = confusion_metrics(concept_label, pred_label)
        accuracy.append((concept_label == pred_label).sum().item()/N)
        if (tp ==0 or (tp + fn)==0):
            recall.append(0)
        else:
            recall.append(tp / (tp + fn))

        if (tn == 0 or (tn + fp)== 0):
            specificity.append(0)
        else:
            specificity.append(tn/(tn + fp))

        if (tp ==0 or (tp + fp)==0) > 0:
            precision.append(0)
        else:
            precision.append(tp/(tp + fp))

        if (precision[-1] + recall[-1]) > 0:
            f1score.append(2*precision[-1]*recall[-1]/(precision[-1] + recall[-1]))
        else:
            f1score.append(0)
    return accuracy, recall, specificity, precision, f1score


def subject_metric(true, pred):
    cat_true = np.concatenate(true)
    cat_pred = np.concatenate(pred)

    accuracy, recall, specificity, f1score, precision = [], [], [], [], []
    N = cat_true.shape[0]

    for i in range(N):
        concept_label = cat_true[i, :]
        pred_label = (cat_pred[i, :] > 0.5).astype(int)
        tn, fp, fn, tp  = confusion_metrics(concept_label, pred_label)
        accuracy.append((concept_label == pred_label).sum().item()/20)
        if (tp ==0 or (tp + fn)==0):
            recall.append(0)
        else:
            recall.append(tp / (tp + fn))

        if (tn == 0 or (tn + fp)== 0):
            specificity.append(0)
        else:
            specificity.append(tn/(tn + fp))

        if (tp ==0 or (tp + fp)==0) > 0:
            precision.append(0)
        else:
            precision.append(tp/(tp + fp))

        if (precision[-1] + recall[-1]) > 0:
            f1score.append(2*precision[-1]*recall[-1]/(precision[-1] + recall[-1]))
        else:
            f1score.append(0)
    return np.mean(accuracy), np.mean(recall), np.mean(specificity), np.mean(precision), np.mean(f1score)


def get_corr(v1, v2):
    ''' v1 and v2 can be matrices of shape (N, d) and (M, d)
    v1 and v2 may or may not must be normalized
    '''
    v1 = F.normalize(v1, dim=1)
    v2 = F.normalize(v2, dim=1)

    return torch.matmul(v1, v2.T) #or size N, M

def model_test_1nn_hbar(device,
             model, model_root, 
             dataloader, label_obj, label_file,
              szdet_thresh=0.45, conceptcorr_thres=0.8, nconc=5):
    sig = nn.Sigmoid() 
    ctrue, cpred = [], []
    strue, spred = [], []
    h_corr_wconc, hsz_corr_wconc, hnsz_corr_wconc = [], [], []
    
    all_emb = label_obj.compconcept_emb
    all_emb = torch.tensor(all_emb).float().to(device)
    for batch_idx, data in enumerate(dataloader):
        inputs = data['buffers']
        det_labels = data['sz_labels'].long().to(device)
        pt_id = data['patient numbers'][0].item()
        concept_labels = label_obj(label_file, pt_id)
        concept_labels = torch.DoubleTensor(concept_labels).reshape( 1,-1).detach().numpy() #.repeat(inputs.shape[1], 1)
        inputs = inputs.to(device)
        hg, sz_pred, _  = model(inputs)
        B, Nsz, T, L = hg.size()
        h_flatten = hg.view(B * Nsz * T, L)
        h_flatten = F.normalize(h_flatten, dim=1)

        det_labels = det_labels.view(-1)
        pred_det = F.softmax(sz_pred.reshape(-1, 2), -1)[:, 1].reshape(-1)
        # Find indices for seizure and non-seizure cases - we hsould use predicted seizure. 
        sz_idx = torch.where(pred_det >= szdet_thresh)[0]
        nsz_idx = torch.where(pred_det < szdet_thresh)[0]
        #assert len(sz_idx)>0 
        
        # Extract embeddings for seizure and non-seizure cases
        if not len(sz_idx) == 0:
            h_sz = h_flatten[sz_idx, :] if len(sz_idx) > 0 else torch.empty(0, h_flatten.size(-1), device=h_flatten.device)
            hbar_sz = h_sz.view(sz_idx.shape[0], L).mean(0).reshape(1, L)
            sz_corr_wconcepts = get_corr(hbar_sz, all_emb).detach().cpu().numpy().reshape(-1)
        else:
            sz_corr_wconcepts = -1 * np.ones(20)
        hsz_corr_wconc.append(sz_corr_wconcepts)

        if not len(nsz_idx) == 0:
            h_nsz = h_flatten[nsz_idx, :] if len(nsz_idx) > 0 else torch.empty(0, h_flatten.size(-1), device=h_flatten.device)
            hbar_nsz = h_nsz.view(nsz_idx.shape[0], L).mean(0).reshape( 1, L)
            nsz_corr_wconcepts = get_corr(hbar_nsz, all_emb).detach().cpu().numpy().reshape(-1)
        else:
            nsz_corr_wconcepts = -1 * np.ones(20)
        hnsz_corr_wconc.append(nsz_corr_wconcepts)

        concept_pred = np.zeros_like(concept_labels)
        concept_pred[0, sz_corr_wconcepts>=conceptcorr_thres] = 1
        concept_pred[0, nsz_corr_wconcepts>=conceptcorr_thres] = 1

        all_h_corr_wconcepts = get_corr(h_flatten, all_emb).detach().cpu().numpy().reshape(-1, 20)
        h_corr_wconc.append(all_h_corr_wconcepts)

        ctrue.append(concept_labels)
        cpred.append(concept_pred)
        spred.append(pred_det.detach().cpu().numpy())
        strue.append(det_labels.view(-1).detach().cpu().numpy())

        
    return ctrue, cpred, strue, spred, h_corr_wconc, hsz_corr_wconc, hnsz_corr_wconc