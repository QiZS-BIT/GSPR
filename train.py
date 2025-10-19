#!/usr/bin/env python3

import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import faiss
import torch
from config.params import ModelParams
from modules.GSPR import GSPR
from modules.loss import triplet_loss
from datasets.nuscenes_dataloader import make_dataloader
from tensorboardX import SummaryWriter

sys.maxsize = 3000
np.set_printoptions(threshold=sys.maxsize)


class Trainer:
    def __init__(self, params: ModelParams):
        super(Trainer, self).__init__()
        self.params = params
        self.weights_root = self.params.weights_root
        self.logs_root = self.params.logs_root
        self.cache_root = self.params.cache_root
        self.resume = self.params.resume_checkpoint
        self.resume_epoch = self.params.resume_epoch
        self.pos_num = params.pos_num
        self.neg_num = params.neg_num

        self.writer = SummaryWriter(self.logs_root)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GSPR()
        n_params = sum([parameter.nelement() for parameter in self.model.parameters()])
        print(f'Number of model parameters: %d' % n_params)
        self.model.to(self.device)
        self.train_dataloader, self.val_dataloader, self.train_query_dataloader, self.val_query_dataloader, self.val_query_dataset \
            = make_dataloader('train', params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.params.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def do_train(self):
        if self.resume:
            if self.params.checkpoint_path:
                resume_filename = self.params.checkpoint_path
            else:
                resume_filename = os.path.join(self.params.weights_root, str(self.params.resume_epoch) + ".pth.tar")
            print("\nResuming from %s" % resume_filename)

            checkpoint = torch.load(resume_filename)
            starting_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print("\nTraining From Scratch")
            starting_epoch = 0

        for i in range(starting_epoch + 1, self.params.epochs + 1):
            print("\nTraining Process-" + "epoch" + " " + str(i) + "-------->")

            # if i >= 50:
            #     self.get_latent_vectors()
            #     self.train_dataloader.dataset.update_latent_vectors(self.cache_root + "/vec_cache.pickle")

            query_num = 0
            total_loss = 0
            self.model.train()
            for batch_idx, current_batch in enumerate(tqdm(self.train_dataloader)):
                gaussian_feat_cuda = current_batch['gaussian_tuple'].to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    descriptors = self.model(gaussian_feat_cuda)
                    o1, o2, o3 = torch.split(descriptors, [1, self.pos_num, self.neg_num], dim=0)
                    loss = triplet_loss(o1, o2, o3, self.params.loss_margin, lazy=True)
                    loss.backward()
                    self.optimizer.step()
                if torch.isnan(loss):
                    raise ValueError("Loss of current batch is nan!")
                query_num = query_num + 1
                total_loss = total_loss + loss.item()
                self.writer.add_scalar('loss', loss.item(), (i-1)*len(self.train_dataloader)+batch_idx)

            loss_cur_epoch = total_loss / query_num
            print(f"epoch %d finishes\n" % i)
            print(f"loss %f\n" % loss_cur_epoch)
            self.writer.add_scalar('total loss', loss_cur_epoch, i)
            self.scheduler.step()

            # save checkpoint status
            weight_filename = os.path.join(self.weights_root, str(i) + ".pth.tar")
            torch.save(
                {
                    'epoch': i,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                },
                weight_filename
            )

            if i % 1 == 0:
                val_loss = self.do_val_loss()
                self.writer.add_scalar('val loss', val_loss, i)

                recall_rate_val = self.do_val()
                self.writer.add_scalar('ValRecall@1', float(recall_rate_val[0]), i*len(self.train_dataloader))
                self.writer.add_scalar('ValRecall@5', float(recall_rate_val[1]), i*len(self.train_dataloader))
                self.writer.add_scalar('ValRecall@10', float(recall_rate_val[2]), i*len(self.train_dataloader))
                self.writer.add_scalar('ValRecall@20', float(recall_rate_val[3]), i*len(self.train_dataloader))
                self.writer.add_scalar('ValRecall@1%', float(recall_rate_val[4]), i*len(self.train_dataloader))

    def get_latent_vectors(self):
        print("\nGenerating Cache-" + "-------->")
        descriptor_database = []
        with torch.no_grad():
            for batch_idx, current_batch in enumerate(tqdm(self.train_query_dataloader)):
                gaussian_feat_cuda = current_batch['gaussian_tuple'].to(self.device)
                self.model.eval()
                descriptor = self.model(gaussian_feat_cuda)
                descriptor_database.append(descriptor[0, :].cpu().detach().numpy())
                torch.cuda.empty_cache()
        vec_filepath = self.cache_root + "/vec_cache.pickle"
        with open(vec_filepath, 'wb') as file:
            pickle.dump(descriptor_database, file)

    def do_val_loss(self):
        total_loss = 0
        query_num = 0
        with torch.no_grad():
            for batch_idx, current_batch in enumerate(tqdm(self.val_dataloader)):
                gaussian_feat_cuda = current_batch['gaussian_tuple'].to(self.device)
                self.model.eval()
                descriptors = self.model(gaussian_feat_cuda)
                o1, o2, o3 = torch.split(descriptors, [1, self.pos_num, self.neg_num], dim=0)
                loss = triplet_loss(o1, o2, o3, self.params.loss_margin, lazy=True)
                torch.cuda.empty_cache()
                total_loss = total_loss + loss.item()
                query_num = query_num + 1
        loss_cur_epoch = total_loss / query_num
        print(f"loss %f\n" % loss_cur_epoch)

        return loss_cur_epoch

    def do_val(self):
        print("\nEvaluating Process-" + "-------->")
        descriptor_query = []
        descriptor_database = []
        gt = self.val_query_dataset.get_positives()

        with torch.no_grad():
            for batch_idx, current_batch in enumerate(tqdm(self.val_query_dataloader)):
                gaussian_feat_cuda = current_batch['gaussian_tuple'].to(self.device)
                self.model.eval()
                descriptor = self.model(gaussian_feat_cuda)
                if batch_idx < self.val_query_dataset.num_db:
                    descriptor_database.append(descriptor[0, :].cpu().detach().numpy())
                else:
                    descriptor_query.append(descriptor[0, :].cpu().detach().numpy())
                torch.cuda.empty_cache()

        total_query_pcd_num = self.val_query_dataset.num_query
        descriptor_query = np.array(descriptor_query).astype('float32')
        descriptor_database = np.array(descriptor_database).astype('float32')
        faiss_index = faiss.IndexFlatL2(self.params.output_dim)
        faiss_index.add(descriptor_database)
        dis, pred_ind = faiss_index.search(descriptor_query, 100)

        top_n = [1, 5, 10, 20, int(total_query_pcd_num * 0.01)]
        successful_recall_num = np.zeros(len(top_n))
        for query_ind in range(total_query_pcd_num):
            for ind, n in enumerate(top_n):
                if np.any(np.in1d(pred_ind[query_ind, :n], np.array(gt[query_ind]))):
                    successful_recall_num[ind:] += 1
                    break

        recall_rate = successful_recall_num / float(total_query_pcd_num) * 100.0
        print(f'\nValRecall@1: %.2f' % float(recall_rate[0]))
        print(f'ValRecall@5: %.2f' % float(recall_rate[1]))
        print(f'ValRecall@10: %.2f' % float(recall_rate[2]))
        print(f'ValRecall@20: %.2f' % float(recall_rate[3]))
        print(f'ValRecall@1percent: %.2f\n' % float(recall_rate[4]))

        return recall_rate


if __name__ == '__main__':
    param_class = ModelParams()
    trainer = Trainer(param_class)
    trainer.do_train()
