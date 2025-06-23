import os
import torch 
import random
import argparse
import numpy as np
from utils import *
from concept_label_creation import *
from Loss import *
from txlstm_szpool import *
from dataloader import *
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from baselines import *

import torch
import torch.nn as nn
import torch.nn.functional as F



class Trainer(nn.Module):
    def __init__(self, cvfold, lr, l1, l2, l3,  name, subdir, pooltype, cplosstype, algnlosstype):
        
        self.cvfold = cvfold
        self.mn_fn = '/project/deepsoz/deeksha/miccai24/data/tuh_single_windowed_manifest.csv'
        self.tuh_root = '/project/deepsoz/deeksha/miccai24/data/'
        self.save_root = '/projectnb/seizuredet/ConceptAlignment/ConceptBottleneckModels/FinalExperiment/Models'
        self.pt_list = np.load(self.tuh_root + 'all_pts.npy')
        self.name = name
        self.subdir = subdir
        
        #initial model and optimizer params
        self.modelname = 'cnnblstm'
        self.traintype = 'pretrain_unc'
        self.oldlr = 0.01
        self.pretrained = (
            f"{self.tuh_root}split{self.cvfold}/models/"
            f"{self.modelname}_{self.traintype}_cv{self.cvfold}_{self.oldlr}.pth.tar"
        )
        self.savename = self.name
        self.tb_writer = SummaryWriter(
            log_dir=os.path.join(
                '/projectnb/seizuredet/ConceptAlignment/ConceptBottleneckModels/FinalExperiment/Tensorboard',
                str(self.cvfold),
                self.subdir, 
                self.savename
            )
        )

        #Initial parameters 
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.pooltype = pooltype
        self.cplosstype = cplosstype
        self.algnlosstype = algnlosstype

    def get_data(self, manifest, obj, label_file):
        train_set =  myDataSet(self.tuh_root, self.cvfold, manifest, normalize=True,train=True)
        val_set = myDataSet(self.tuh_root, self.cvfold, manifest, normalize=True,train=False)
        #train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
        class_weights = self.count_class_weights(label_file, obj)  #1-norm_count, compare with 1/count
        samples_weight = []
        for sample in train_set.mnlist:
            pt_id = np.int64(sample['pt_id'])
            target = obj(label_file, pt_id).reshape(-1)
            weight = max(class_weights[target==1]) #min if count, not weights
            samples_weight.append(weight)
        samples_weight = torch.DoubleTensor(samples_weight)
        #samples_weight = 1/samples_weight
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight)//3,  replacement = True)
        train_loader = DataLoader(train_set, batch_size=1, sampler=sampler, num_workers=2 )

        val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=2)
        return train_loader, val_loader

    def train_tensorboard(self, epoch, epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss):
        self.tb_writer.add_scalar('concept prediction/train', torch.tensor(epoch_loss1), epoch)
        self.tb_writer.add_scalar('alignment/train', torch.tensor(epoch_loss2), epoch)
        self.tb_writer.add_scalar('l2 penalty/train', torch.tensor(epoch_loss3), epoch)
        self.tb_writer.add_scalar('train loss', torch.tensor(epoch_loss), epoch)

    def val_tensorboard(self, epoch, val_epoch_loss1, val_epoch_loss2, val_epoch_loss3, val_epoch_loss):
        self.tb_writer.add_scalar('concept prediction/validation', torch.tensor(val_epoch_loss1), epoch)
        self.tb_writer.add_scalar('alignment/validation', torch.tensor(val_epoch_loss2), epoch)
        self.tb_writer.add_scalar('l2 penalty/validation', torch.tensor(val_epoch_loss3), epoch)
        self.tb_writer.add_scalar('validation loss', torch.tensor(val_epoch_loss), epoch)




    def save_concept_file(self, save_path, label_generator):
        concpt_file = label_generator.get_data()
        conf_path = os.path.join(save_path, 'concepts.csv')
        concpt_file.to_csv(conf_path, index=False)

    def count_class_weights(self, labelcsv, label_generator, eps=0.01):
        
        counts = np.zeros(20)
        for j in range(len(labelcsv)):
        
            concept_labels = label_generator(labelcsv, labelcsv['pt_id'][j])
            #concept_labels, sz_emb, nsz_emb = label_generator(pt_id)
            counts += concept_labels.reshape(-1)
        counts /= len(labelcsv)
        counts = 1 - eps - counts 
        counts[counts <= 0] = eps
        return torch.tensor(counts)

    
    def run_fold_CP(self):    
        save_path = os.path.join(self.save_root, f'split{self.cvfold}', self.subdir, self.savename)
        os.makedirs(save_path, exist_ok=True)
        
        #label_generator = concept_label_creator(gpu_id='cuda:1',  prompts_csv = 'prompts.csv')
        #self.save_concept_file(save_path, label_generator)
        label_file = pd.read_csv(os.path.join(os.path.join(self.save_root, f'split{self.cvfold}', self.subdir), 'concepts.csv'))
        label_generator = label_generation()
        print('Label done')
        
        device = 'cuda:0'
        manifest = read_manifest(self.mn_fn, ',')
        train_loader, val_loader = self.get_data(manifest, label_generator, label_file)
        all_emb = torch.tensor(label_generator.compconcept_emb).double().to(device) 
        print('Data loaded')

        #define loss function 

        alignment_loss = SupContLoss_general(self.algnlosstype, all_emb)

        if self.cplosstype == 'bce':
            weights = 0.5*torch.ones(20).to(device)
            concept_pred_loss = MultiLabelBCELoss(alpha = weights)
        elif self.cplosstype == 'wbce':
            weights = self.count_class_weights(label_file, label_generator).to(device)
            concept_pred_loss = MultiLabelBCELoss(alpha = weights) 
        elif self.cplosstype == 'focal':
            weights = 0.5*torch.ones(20).to(device)
            concept_pred_loss = MultiLabelFocalLoss(alpha = weights)
        elif self.cplosstype == 'wfocal':
            weights = self.count_class_weights(label_file, label_generator).to(device)
            concept_pred_loss = MultiLabelFocalLoss(alpha = weights)
        else:
            raise ValueError("Unknown loss type")
        #Initiate model
        model = CNNBLSTM_wcorrBN_CP_2(all_emb = all_emb, pooltype=self.pooltype, device =  'cuda:0' ) 
        model.load_state_dict(torch.load(self.pretrained), strict=False)
        model.double()
        model.to(device)
        #freeze weights
        for name, param in model.named_parameters():
            if "concept" not in name:                    
                param.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        
        #train loop
        sig = nn.Sigmoid()
        for epoch in range(1, 201):
            train_len = len(train_loader)
            val_len = len(val_loader)

            #zero out loss value
            epoch_loss = 0.0
            val_epoch_loss = 0.0
            epoch_loss1, epoch_loss2, epoch_loss3 = 0.0, 0.0, 0.0
            val_epoch_loss1, val_epoch_loss2, val_epoch_loss3 = 0.0, 0.0, 0.0
            model.train()
            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = data['buffers']
                inputs = inputs.to(torch.DoubleTensor()).to(device)

                det_labels = data['sz_labels'].long().to(device)
                pt_id = data['patient numbers'][0].item()
                #concept_labels, sz_emb, nsz_emb = label_generator(pt_id)
                
                hgcorr, hg, sz_pred, concept_pred, _  = model(inputs) 

                concept_labels= label_generator(label_file, pt_id)
                loss2 = alignment_loss(hg, hgcorr, concept_labels, det_labels)

                concept_labels = torch.DoubleTensor(concept_labels).to(device).reshape(1, -1).repeat(inputs.shape[0]*inputs.shape[1], 1)
                loss1 = concept_pred_loss(sig(concept_pred.reshape(-1, 20)), concept_labels.reshape(-1, 20))

                loss3 = sum(p.pow(2.0).sum() for p in model.concept_linear.parameters()) + sum(p.pow(2.0).sum() for p in model.concept_pred_linear.parameters())
                
                loss = self.l1 * loss1 + self.l2 * loss2 + self.l3 * loss3

                
                loss.backward()
                optimizer.step()
                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()
                epoch_loss3 +=  loss3.item()
                epoch_loss += loss.item()  

            print('Train done')

            self.train_tensorboard(epoch, epoch_loss1/train_len, epoch_loss2/train_len, epoch_loss3/train_len, epoch_loss/train_len)
            model.eval()
            for batch_idx, data in enumerate(val_loader):
                optimizer.zero_grad()
                with torch.no_grad():
                    inputs = data['buffers']
                    inputs = inputs.to(torch.DoubleTensor()).to(device)

                    det_labels = data['sz_labels'].long().to(device)
                    pt_id = data['patient numbers'][0].item()
                    #concept_labels, sz_emb, nsz_emb = label_generator(pt_id)
                    hgcorr, hg, sz_pred, concept_pred, _  = model(inputs) 

                    concept_labels= label_generator(label_file, pt_id)
                    loss2 = alignment_loss(hg, hgcorr, concept_labels, det_labels)

                    concept_labels = torch.DoubleTensor(concept_labels).to(device).reshape(1, -1).repeat(inputs.shape[0]*inputs.shape[1], 1)
                    loss1 = concept_pred_loss(sig(concept_pred.reshape(-1, 20)), concept_labels.reshape(-1, 20))

                    loss3 = sum(p.pow(2.0).sum() for p in model.concept_linear.parameters()) + sum(p.pow(2.0).sum() for p in model.concept_pred_linear.parameters())
                    
                    loss = self.l1 * loss1 + self.l2 * loss2 + self.l3 * loss3

                    val_epoch_loss1 +=  loss1.item()
                    val_epoch_loss2 +=  loss2.item()
                    val_epoch_loss3 +=  loss3.item()
                    val_epoch_loss += loss.item()
                    

            print('Val done')
            self.val_tensorboard(epoch, val_epoch_loss1/val_len, val_epoch_loss2/val_len, val_epoch_loss3/val_len, val_epoch_loss/val_len)
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pth.tar'))

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description="Run real_data_crossval with command line inputs.")

    parser.add_argument('--cvfold', type=int, required=True, help='')
    parser.add_argument('--lr', type=float, required=True, help='learning rate')
    parser.add_argument('--l1', type=float, required=True, help='l1')
    parser.add_argument('--l2', type=float, required=True, help='l2')
    parser.add_argument('--l3', type=float, required=True, help='l3')
    parser.add_argument('--subdir', type=str, required=True, help='tuning phase')
    parser.add_argument('--pooltype', type=str, required=True, help='model pool layer')
    parser.add_argument('--cplosstype', type=str, required=True, help='CP loss criteria')
    parser.add_argument('--algnlosstype',  type=str, nargs='+', required=True, help='alignment loss type')


    args = parser.parse_args()

    lr = args.lr
    lr_name = f"{lr:.0e}"
    l1 = args.l1
    l2 = args.l2
    l3 = args.l3

    lr_name = f"{lr:.0e}"
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = f'lr_{lr_name}_l1_{l1}_l2_{l2}_l3_{l3}__Pool:{args.pooltype}_CP:{args.cplosstype}_Align:{args.algnlosstype[0]} {args.algnlosstype[1]}'
    
    trainer = Trainer(args.cvfold, lr, l1, l2, l3, name, args.subdir, args.pooltype, args.cplosstype, (args.algnlosstype[0], args.algnlosstype[1]))
    trainer.run_fold_CP()

# for l1 in 1; do for l2 in 0 0.5 1; do for l3 in 0 0.01 0.001;
# do for lr in 1e-04 1e-05;
# do for pooltype in 'attnAvg' 'avg' 'max' 'attnMax'
# do for cplosstype in 'bce' 'wbce';
# do for algnlosstype in 'tp_average only_cont_nd_sumout' 
# do export l1 l2 l3 lr pooltype cplosstype algnlosstype;  
# qsub -V run_cnn.sh; done; done; done; done; done; done; done


# for l1 in 1 ; do for l2 in 1 ; do for l3 in 1;
# do for lr in 1e-04;
# do for pooltype in 'attnAvg'
# do for cplosstype in 'bce';
# do for algnlosstype in 'tp_average only_cont_nd_sumout' 
# do export l1 l2 l3 lr pooltype cplosstype algnlosstype;  
# qsub -V run_m.sh; done; done; done; done; done; done; done