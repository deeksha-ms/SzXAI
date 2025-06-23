
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


        
import numpy
class MultiLabelBCELoss(nn.Module):
    def __init__(self, alpha=1, gamma=None, reduction='mean'):
        """
        Initialize Focal Loss.

        Args:
        - alpha (float): Balancing factor for class imbalance (default=1).
        - gamma (float): Focusing parameter to reduce loss for well-classified samples (default=2).
        - reduction (str): How to reduce the loss: 'mean', 'sum', or 'none'.
        """
        super(MultiLabelBCELoss, self).__init__()
        self.alpha = alpha.reshape(1, -1)
        self.reduction = reduction
        self.gamma = gamma
    def cross_entropy(self, p, y):
        return -1* y * torch.log(p)

    def forward(self, inputs, targets):
        """
        Compute Focal Loss.

        Args:
        - inputs (Tensor): Predicted probabilities (after sigmoid/softmax). (B, Nk)
        - targets (Tensor): Ground truth labels (binary for BCE, one-hot or class indices for softmax). (B, Nk)

        Returns:
        - Tensor: Computed Focal Loss.
        """
        # Ensure inputs are probabilities (if not already applied)
        weightsum = self.alpha.sum() + 1e-03
        wbceloss =  self.alpha     * self.cross_entropy(inputs, targets) 
        wbceloss += (1-self.alpha) * self.cross_entropy(1-inputs, 1-targets) 
        wbceloss = wbceloss.sum(-1) / weightsum
        # Apply reduction
        if self.reduction == 'mean':
            return wbceloss.mean()
        elif self.reduction == 'sum':
            return wbceloss.sum()
        else:
            return wbceloss

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Initialize Focal Loss.

        Args:
        - alpha (float): Balancing factor for class imbalance (default=1).
        - gamma (float): Focusing parameter to reduce loss for well-classified samples (default=2).
        - reduction (str): How to reduce the loss: 'mean', 'sum', or 'none'.
        """
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha.reshape(1, -1)
        self.gamma = gamma
        self.reduction = reduction
    def cross_entropy(self, p, y):
        return -1* y * torch.log(p)

    def forward(self, inputs, targets):
        """
        Compute Focal Loss.

        Args:
        - inputs (Tensor): Predicted probabilities (after sigmoid/softmax). (B, Nk)
        - targets (Tensor): Ground truth labels (binary for BCE, one-hot or class indices for softmax). (B, Nk)

        Returns:
        - Tensor: Computed Focal Loss.
        """
        # Ensure inputs are probabilities (if not already applied)
        weightsum = self.alpha.sum() + 1e-03
        wF_loss =  self.alpha     * (1-inputs)**self.gamma * self.cross_entropy(inputs, targets) 
        wF_loss += (1-self.alpha) * (inputs)**self.gamma * self.cross_entropy(1-inputs, 1-targets) 
        wF_loss = wF_loss.sum(-1) / weightsum
        # Apply reduction
        if self.reduction == 'mean':
            return wF_loss.mean()
        elif self.reduction == 'sum':
            return wF_loss.sum()
        else:
            return wF_loss

class SupContLoss_general(nn.Module):
    def __init__(self, losstype, all_emb, threshold=0.5, temperature=0.07):
        super(SupContLoss_general, self).__init__()
        self.temperature = temperature
        self.losstype = losstype
        self.all_emb = all_emb
        self.threshold = torch.full((20,), threshold)

    def get_mean(self, hg, det_labels):
        B, Nsz, T, L = hg.size()
        Y = det_labels.view(-1)
        h_flatten = hg.view(B * Nsz * T, L)

        # Find indices for seizure and non-seizure cases
        sz_idx = torch.where(Y == 1)[0]
        nsz_idx = torch.where(Y == 0)[0]

        #Get mean over sz timepoints and nonsz timepoints
        h_bar_sz = torch.mean(h_flatten[sz_idx, :], dim = 0).unsqueeze(0)
        h_bar_sz = F.normalize(h_bar_sz, dim=1)
        h_bar_nsz = torch.mean(h_flatten[nsz_idx, :], dim = 0).unsqueeze(0)
        h_bar_nsz = F.normalize(h_bar_nsz, dim=1)

        return h_bar_sz, h_bar_nsz, sz_idx, nsz_idx

    def compute_loss_type1(self, hsz, hnsz, concept_labels, Psz_idx, Pnsz_idx):

        '''
        loss equation:
        szloss = 1/Psz sum_{j = 1}^Psz(-log(exp(<h_bar_sz, e_j>)/sum_{a in Pa/j} exp(<h_bar_sz, e_a>)))
        nszloss = 1/Pnsz sum_{j = 1}^Pnsz(-log(exp(<h_bar_nsz, e_j>)/sum_{a in Pa/j} exp(<h_bar_nsz, e_a>)))
        '''

        #Get similarities between h_bar_sz and all concepts & h_bar_nsz and all concepts
        sz_sim = torch.exp(torch.matmul(hsz, self.all_emb.T) / self.temperature) #(1, 20)
        nsz_sim = torch.exp(torch.matmul(hnsz, self.all_emb.T) / self.temperature) #(1, 20)
        all_indices = np.arange(len(concept_labels))

        if Psz_idx.shape[0] > 0:
            excluded_indices = np.delete(all_indices, Psz_idx) 
            denominator = torch.sum(sz_sim[:, excluded_indices])
            sz_loss = torch.mean(-torch.log(sz_sim[:, Psz_idx]/denominator))
        else:
            sz_loss = -torch.log(1/torch.sum(sz_sim))

        if Pnsz_idx.shape[0] > 0:
            excluded_indices = np.delete(all_indices, Pnsz_idx) 
            denominator = torch.sum(nsz_sim[:, excluded_indices])

            nsz_loss = torch.mean(-torch.log(nsz_sim[:, Pnsz_idx]/denominator))
        else:
            nsz_loss = -torch.log(1/torch.sum(nsz_sim))

        # Combine losses
        loss = sz_loss + nsz_loss
        return loss

    def compute_loss_type2(self, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx):
        device = hg_corr_econ.device
        all_sim = torch.exp(hg_corr_econ / self.temperature)
        sz_sim = all_sim[sz_idx, :]
        nsz_sim = all_sim[nsz_idx, :]

        sz_loss, nsz_loss = torch.tensor([0], dtype=torch.double).to(device), torch.tensor([0], dtype=torch.double).to(device)
        all_indices = np.arange(len(concept_labels))

        # Compute loss components
        if Psz_idx.shape[0] > 0:
            for i, isz in enumerate(Psz_idx):
                excluded_indices = np.delete(all_indices, isz)
                excluded_sim = sz_sim[:, excluded_indices]
                th = self.threshold[isz]
                column = sz_sim[:, isz]
                ktop_indices = (column > th).nonzero(as_tuple=True)
                nominator = column[ktop_indices]
                if nominator.shape[0] == 0:
                    continue
                denominator = torch.sum(excluded_sim[ktop_indices[0], :], dim = 1)
                sz_loss += torch.mean(-torch.log(nominator/denominator))
            sz_loss = sz_loss/Psz_idx.shape[0]
        else:
            nsz_conc_sztp_sim = sz_sim[:, 10:]
            sz_loss = torch.mean(-torch.log(1/torch.sum(nsz_conc_sztp_sim, dim=1)))

        if Pnsz_idx.shape[0] > 0:
            for i, insz in enumerate(Pnsz_idx):
                excluded_indices = np.delete(all_indices, insz)
                excluded_sim = nsz_sim[:, excluded_indices]
                th = self.threshold[insz]
                column = nsz_sim[:, insz]
                ktop_indices = (column > th).nonzero(as_tuple=True)
                nominator = column[ktop_indices]
                if nominator.shape[0] == 0:
                    continue
                denominator = torch.sum(excluded_sim[ktop_indices[0], :], dim = 1)
                nsz_loss += torch.mean(-torch.log(nominator/denominator))
            
            nsz_loss = nsz_loss/Pnsz_idx.shape[0]
        else:
            sz_conc_nsztp_sim = nsz_sim[:, :10]
            nsz_loss = torch.mean(-torch.log(1/torch.sum(sz_conc_nsztp_sim, dim=1)))

        # Combine losses
        loss = sz_loss + nsz_loss

        return loss
    
    def compute_loss_type3(self, hg, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx):
        B, Nsz, T, L = hg.size()
        h_flatten = hg.view(B * Nsz * T, L)
        h_flatten = F.normalize(h_flatten, dim=1)
        hsz = h_flatten[sz_idx, :]
        hnsz = h_flatten[nsz_idx, :]

        all_sim = torch.exp(hg_corr_econ / self.temperature)
        sz_sim = all_sim[sz_idx, :]
        nsz_sim = all_sim[nsz_idx, :]

        all_indices = np.arange(len(concept_labels))
        if Psz_idx.shape[0] > 0:
            e_psz_bar = torch.mean(self.all_emb[Psz_idx, :], dim = 0).unsqueeze(0)
            excluded_indices = np.delete(all_indices, Psz_idx) 
            nominator = torch.exp(torch.matmul(hsz, e_psz_bar.T)/self.temperature)
            denominator = torch.sum(sz_sim[:, excluded_indices], dim = 1)
            sz_loss = torch.mean(-torch.log(nominator/denominator))
        else:
            sz_loss = torch.mean(-torch.log(1/torch.sum(sz_sim, dim = 1)))

        if Pnsz_idx.shape[0] > 0:
            e_pnsz_bar = torch.mean(self.all_emb[Pnsz_idx, :], dim = 0).unsqueeze(0)
            excluded_indices = np.delete(all_indices, Pnsz_idx) 
            nominator = torch.exp(torch.matmul(hnsz, e_pnsz_bar.T)/self.temperature)
            denominator = torch.sum(nsz_sim[:, excluded_indices], dim = 1)
            nsz_loss = torch.mean(-torch.log(nominator/denominator))
        else:
            nsz_loss = torch.mean(-torch.log(1/torch.sum(nsz_sim, dim = 1)))

        loss = sz_loss + nsz_loss

        return loss

    def compute_loss_type4(self, hg, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx):
        B, Nsz, T, L = hg.size()
        h_flatten = hg.view(B * Nsz * T, L)
        h_flatten = F.normalize(h_flatten, dim=1)
        hsz = h_flatten[sz_idx, :]
        hnsz = h_flatten[nsz_idx, :]

        all_sim = torch.exp(hg_corr_econ / self.temperature)
        sz_sim = all_sim[sz_idx, :]
        nsz_sim = all_sim[nsz_idx, :]

        all_indices = np.arange(len(concept_labels))
        if Psz_idx.shape[0] > 0:
            e_psz_bar = torch.mean(self.all_emb[Psz_idx, :], dim = 0).unsqueeze(0)
            nominator = torch.exp(torch.matmul(hsz, e_psz_bar.T)/self.temperature)
            denominator = torch.sum(sz_sim[:, 10:], dim = 1)
            sz_loss = torch.mean(-torch.log(nominator/denominator))
        else:
            sz_loss = torch.mean(-torch.log(1/torch.sum(sz_sim[:, 10:], dim = 1)))

        if Pnsz_idx.shape[0] > 0:
            e_pnsz_bar = torch.mean(self.all_emb[Pnsz_idx, :], dim = 0).unsqueeze(0)
            nominator = torch.exp(torch.matmul(hnsz, e_pnsz_bar.T)/self.temperature)
            denominator = torch.sum(nsz_sim[:, :10], dim = 1)
            nsz_loss = torch.mean(-torch.log(nominator/denominator))
        else:
            nsz_loss = torch.mean(-torch.log(1/torch.sum(nsz_sim[:, :10], dim = 1)))

        loss = sz_loss + nsz_loss

        return loss

    def compute_loss_type5(self, hg, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx):
        B, Nsz, T, L = hg.size()
        h_flatten = hg.view(B * Nsz * T, L)
        h_flatten = F.normalize(h_flatten, dim=1)
        hsz = h_flatten[sz_idx, :]
        hnsz = h_flatten[nsz_idx, :]

        all_sim = torch.exp(hg_corr_econ / self.temperature)
        sz_sim = all_sim[sz_idx, :]
        nsz_sim = all_sim[nsz_idx, :]

        all_indices = np.arange(len(concept_labels))
        if Psz_idx.shape[0] > 0:
            nominator = torch.sum(sz_sim[:, Psz_idx], dim=1)
            denominator = torch.sum(sz_sim[:, 10:], dim = 1)
            sz_loss = torch.mean(-torch.log(nominator/denominator))
        else:
            sz_loss = torch.mean(-torch.log(1/torch.sum(sz_sim[:, 10:], dim = 1)))

        if Pnsz_idx.shape[0] > 0:
            nominator = torch.sum(nsz_sim[:, Pnsz_idx], dim=1)
            denominator = torch.sum(nsz_sim[:, :10], dim = 1)
            nsz_loss = torch.mean(-torch.log(nominator/denominator))
        else:
            nsz_loss = torch.mean(-torch.log(1/torch.sum(nsz_sim[:, :10], dim = 1)))

        loss = sz_loss + nsz_loss

        return loss

    def compute_loss_type6(self, hg, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx):
        B, Nsz, T, L = hg.size()
        h_flatten = hg.view(B * Nsz * T, L)
        h_flatten = F.normalize(h_flatten, dim=1)
        hsz = h_flatten[sz_idx, :]
        hnsz = h_flatten[nsz_idx, :]

        all_sim = torch.exp(hg_corr_econ / self.temperature)
        sz_sim = all_sim[sz_idx, :]
        nsz_sim = all_sim[nsz_idx, :]

        all_indices = np.arange(len(concept_labels))
        if Psz_idx.shape[0] > 0:
            nominator = sz_sim[:, Psz_idx]
            denominator = torch.sum(sz_sim[:, 10:], dim = 1).unsqueeze(1)
            sz_loss = torch.mean(-torch.log(nominator/denominator))
        else:
            sz_loss = torch.mean(-torch.log(1/torch.sum(sz_sim[:, 10:], dim = 1)))

        if Pnsz_idx.shape[0] > 0:
            nominator = nsz_sim[:, Pnsz_idx]
            denominator = torch.sum(nsz_sim[:, :10], dim = 1).unsqueeze(1)
            nsz_loss = torch.mean(-torch.log(nominator/denominator))
        else:
            nsz_loss = torch.mean(-torch.log(1/torch.sum(nsz_sim[:, :10], dim = 1)))

        loss = sz_loss + nsz_loss

        return loss

    def compute_loss_type7(self, hg, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx):
        B, Nsz, T, L = hg.size()
        h_flatten = hg.view(B * Nsz * T, L)
        h_flatten = F.normalize(h_flatten, dim=1)
        hsz = h_flatten[sz_idx, :]
        hnsz = h_flatten[nsz_idx, :]

        all_sim = torch.exp(hg_corr_econ / self.temperature)
        sz_sim = all_sim[sz_idx, :]
        nsz_sim = all_sim[nsz_idx, :]

        all_indices = np.arange(len(concept_labels))
        if Psz_idx.shape[0] > 0:
            nominator = torch.mean(sz_sim[:, Psz_idx], dim=1)
            denominator = torch.sum(sz_sim[:, 10:], dim = 1)
            sz_loss = torch.mean(-torch.log(nominator/denominator))
        else:
            sz_loss = torch.mean(-torch.log(1/torch.sum(sz_sim[:, 10:], dim = 1)))

        if Pnsz_idx.shape[0] > 0:
            nominator = torch.mean(nsz_sim[:, Pnsz_idx], dim=1)
            denominator = torch.sum(nsz_sim[:, :10], dim = 1)
            nsz_loss = torch.mean(-torch.log(nominator/denominator))
        else:
            nsz_loss = torch.mean(-torch.log(1/torch.sum(nsz_sim[:, :10], dim = 1)))

        loss = sz_loss + nsz_loss

        return loss
    
    def forward(self, hg, hg_corr_econ, concept_labels, det_labels):
        Psz_idx = np.where(concept_labels[:10] == 1)[0]
        Pnsz_idx = 10+np.where(concept_labels[10:] == 1)[0]

        h_bar_sz, h_bar_nsz, sz_idx, nsz_idx = self.get_mean(hg, det_labels)
        
        if self.losstype == ('concept_average', 'average_htilde'):
            loss = self.compute_loss_type1(h_bar_sz, h_bar_nsz, concept_labels, Psz_idx, Pnsz_idx)
        
        if self.losstype == ('concept_average', 'ktop'):
            loss = self.compute_loss_type2(hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx)

        if self.losstype == ('tp_average', 'all_nonpresent_conc'):
            loss = self.compute_loss_type3(hg, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx)

        if self.losstype == ('tp_average', 'only_cont_conc'):
            loss = self.compute_loss_type4(hg, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx)

        if self.losstype == ('tp_average', 'only_cont_nd'):
            loss = self.compute_loss_type5(hg, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx)
        if self.losstype == ('tp_average', 'only_cont_nd_sumout'):
            loss = self.compute_loss_type6(hg, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx)
        if self.losstype == ('tp_average', 'only_cont_nd_meanin'):
            loss = self.compute_loss_type7(hg, hg_corr_econ, concept_labels, sz_idx, nsz_idx, Psz_idx, Pnsz_idx)
        return loss



# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, h_g, e_s, e_ns, label):
#         device = h_g.device
#         B, Nsz, T, L = h_g.size()

#         # Flatten tensors
#         Y = label.view(-1)
#         h_flatten = h_g.view(B * Nsz * T, L)
#         h_flatten = F.normalize(h_flatten, dim=1)

#         # Find indices for seizure and non-seizure cases
#         sz_idx = torch.where(Y == 1)[0]
#         nsz_idx = torch.where(Y == 0)[0]

#         # Extract embeddings for seizure and non-seizure cases
#         h_sz = h_flatten[sz_idx, :] if len(sz_idx) > 0 else torch.empty(0, h_flatten.size(-1), device=h_flatten.device)
#         h_nsz = h_flatten[nsz_idx, :] if len(nsz_idx) > 0 else torch.empty(0, h_flatten.size(-1), device=h_flatten.device)

#         # Compute similarities
#         sz_sim = torch.exp(torch.matmul(h_flatten, e_s.T) / self.temperature)
#         nsz_sim = torch.exp(torch.matmul(h_flatten, e_ns.T) / self.temperature)
#         denominator = sz_sim + nsz_sim

#         # Compute loss components
#         if len(sz_idx) > 0:
#             h_sz_sim = torch.exp(torch.matmul(h_sz, e_s.T) / self.temperature)
#             # sz_loss = -torch.log(h_sz_sim / denominator[sz_idx]).mean()
#             sz_loss = -torch.log((h_sz_sim*sz_idx.shape[0]) / (sz_sim[sz_idx]*sz_idx.shape[0] + nsz_sim[sz_idx]*nsz_idx.shape[0])).mean()
#         else:
#             sz_loss = 0

#         if len(nsz_idx) > 0:
#             h_nsz_sim = torch.exp(torch.matmul(h_nsz, e_ns.T) / self.temperature)
#             # nsz_loss = -torch.log(h_nsz_sim / denominator[nsz_idx]).mean()
#             nsz_loss = -torch.log((h_nsz_sim*nsz_idx.shape[0]) / (sz_sim[nsz_idx]*sz_idx.shape[0] + nsz_sim[nsz_idx]*nsz_idx.shape[0])).mean()
#         else:
#             nsz_loss = 0

#         # Combine losses
#         loss = sz_loss + nsz_loss

#         return loss, sz_sim[sz_idx].mean().item(), nsz_sim[nsz_idx].mean().item(), nsz_sim[sz_idx].mean().item(), sz_sim[nsz_idx].mean().item()


