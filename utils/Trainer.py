import os
import time

import numpy as np
import tqdm
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.metrics import *
import torch
from torch import nn

class Trainer():
    def __init__(self,
                model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer,
                 epoch_stop,
                 epoches,
                 save_threshold = 0.0,
                 start_epoch = 0,
                 lambd = 1.0,
                 ):
        
        self.model = model
        self.device = device
        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer


        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            os.makedirs(save_param_path, exist_ok=True)
            self.save_param_path = save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout=dropout
        self.lambd = lambd

        self.criterion = nn.CrossEntropyLoss()
        self._phase_results = {}

        # Curriculum learning: epochs 分界点和对应的训练比例
        self.curriculum_epochs = [10, 20]
        self.curriculum_ratios = [0.3, 0.7, 1.0]

        num_train_batches = len(self.dataloaders['train'])
        print(f"Total training batches: {num_train_batches}")
        print(f"Curriculum learning: epochs 1-{self.curriculum_epochs[0]} -> {int(self.curriculum_ratios[0]*100)}% of batches")
        print(f"Curriculum learning: epochs {self.curriculum_epochs[0]+1}-{self.curriculum_epochs[1]} -> {int(self.curriculum_ratios[1]*100)}% of batches")
        print(f"Curriculum learning: epochs {self.curriculum_epochs[1]+1}+ -> {int(self.curriculum_ratios[2]*100)}% of batches")

    def train(self):
        since = time.time()
        self.model.cuda()
        best_acc_val = 0.0
        best_epoch_val = 0
        is_earlystop = False
        last_save_path = ''

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)

            # 更新学习率
            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75

            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

            current_epoch = epoch - self.start_epoch
            if current_epoch < self.curriculum_epochs[0]:
                ratio_label = self.curriculum_ratios[0]
            elif current_epoch < self.curriculum_epochs[1]:
                ratio_label = self.curriculum_ratios[1]
            else:
                ratio_label = self.curriculum_ratios[2]
            print(f'[Reverse Curriculum Learning] Epoch {epoch+1}: using last {int(ratio_label*100)}% training data')

            for phase in ['train', 'val','test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                print('-' * 10)
                print (phase.upper())
                print('-' * 10)

                running_loss = 0.0
                running_loss_recon = 0.0
                tpred = []
                tlabel = []
                num_samples = 0

                if phase == 'train':
                    current_epoch = epoch - self.start_epoch
                    if current_epoch < self.curriculum_epochs[0]:
                        ratio = self.curriculum_ratios[0]
                    elif current_epoch < self.curriculum_epochs[1]:
                        ratio = self.curriculum_ratios[1]
                    else:
                        ratio = self.curriculum_ratios[2]
                    
                    all_batches = list(self.dataloaders['train'])
                    num_samples = max(1, int(len(all_batches) * ratio))
                    indices = np.random.choice(len(all_batches), size=num_samples, replace=False)
                    data_iter = iter([all_batches[i] for i in indices])
                    print(f'Using {num_samples}/{len(all_batches)} batches ({ratio*100:.0f}%)')
                else:
                    data_iter = iter(self.dataloaders[phase])

                for batch in tqdm(data_iter):
                    batch_data=batch
                    # to gpu
                    for k,v in batch_data.items():
                        if k == 'missing_type':
                            continue  # list of strings, no need to cuda
                        batch_data[k]=v.cuda()
                    label = batch_data['label']

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, loss_recon = self.model(**batch_data)
                        _, preds = torch.max(outputs, 1)
                        loss =0.1* self.criterion(outputs, label) +0.1* self.lambd * loss_recon
                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)
                    running_loss_recon += loss_recon.item() * label.size(0)
                    num_samples += label.size(0)
                    
                epoch_loss = running_loss / num_samples
                epoch_loss_recon = running_loss_recon / num_samples
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print (results)
                get_confusionmatrix_fnd(tpred,tlabel)

                self.writer.add_scalar('Loss/'+phase, epoch_loss, epoch+1)
                self.writer.add_scalar('Loss_recon/'+phase, epoch_loss_recon, epoch+1)
                self.writer.add_scalar('Acc/'+phase, results['acc'], epoch+1)
                self.writer.add_scalar('F1/'+phase, results['f1'], epoch+1)

                if phase not in self._phase_results:
                    self._phase_results[phase] = {'acc': [], 'f1': [], 'precision': [], 'recall': []}
                self._phase_results[phase]['acc'].append(results['acc'])
                self._phase_results[phase]['f1'].append(results['f1'])
                self._phase_results[phase]['precision'].append(results['precision'])
                self._phase_results[phase]['recall'].append(results['recall'])
                
                
                if phase == 'val' and results['acc'] > best_acc_val:
                    best_acc_val = results['acc']
                    best_epoch_val = epoch + 1
                    if best_acc_val > self.save_threshold:
                        if os.path.exists(last_save_path):
                            print('delete the previous checkpoint...')
                            os.remove(last_save_path)
                            save_path = self.save_param_path + "_test_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val)
                            torch.save(self.model.state_dict(),save_path)
                            last_save_path = save_path
                            print("saved " + self.save_param_path + "_test_epoch" + str(
                                best_epoch_val) + "_{0:.4f}".format(best_acc_val))
                    else:
                        if epoch - best_epoch_val >= self.epoch_stop - 1:
                            is_earlystop = True
                            print("early stopping...")
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))

        final_results = {}
        for phase, records in self._phase_results.items():
            if phase == 'val' and len(records['acc']) > 0:
                best_epoch_idx = np.argmax(records['acc'])
            else:
                best_epoch_idx = -1
            final_results[phase] = {
                'acc': records['acc'][best_epoch_idx],
                'f1': records['f1'][best_epoch_idx],
                'precision': records['precision'][best_epoch_idx],
                'recall': records['recall'][best_epoch_idx],
            }
            print(f"Best {phase} (epoch {best_epoch_idx+1}): acc={final_results[phase]['acc']:.4f}, f1={final_results[phase]['f1']:.4f}")

        return final_results


    
