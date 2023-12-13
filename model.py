import os
import logging
import numpy as np
import torch
import math
import random
from torch.autograd import Variable
from tqdm import tqdm
from combination import ClickModel
from tensorboardX import SummaryWriter
from torch import nn

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
MINF = 1e-30
# device = torch.device('cpu')

class Model(object):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        self.args = args
        self.logger = logging.getLogger("TRACM")
        self.eval_freq = args.eval_freq
        self.learning_rate = args.learning_rate
        self.global_step = args.load_model if args.load_model > -1 else 0
        self.patience = args.patience
        self.writer = None
        if args.train:
            self.writer = SummaryWriter(self.args.summary_dir)

        self.model = ClickModel(self.args, query_size, doc_size, vtype_size, dataset)
        if args.data_parallel:
            self.model = nn.DataParallel(self.model)
        if use_cuda:
            self.model = self.model.cuda()
        self.optimizer = self.create_train_op()
        self.loss_criterion = nn.BCELoss()

        # NDCG Truncation Levels
        self.trunc_levels = [1, 3, 5, 10]

    def compute_loss(self, pred_logits, TRUE_CLICKS):
        """
        The loss function
        """
        return self.loss_criterion(pred_logits, TRUE_CLICKS)

    def compute_perplexity(self, pred_logits, TRUE_CLICKS):
        '''
        Compute the perplexity
        '''
        pos_logits = torch.log2(pred_logits + MINF)
        neg_logits = torch.log2(1. - pred_logits + MINF)
        perplexity_at_rank = torch.where(TRUE_CLICKS == 1, pos_logits, neg_logits).sum(dim=0)
        return perplexity_at_rank

    def create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.args.optim == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            weight_decay=self.args.weight_decay)
        elif self.args.optim == 'adadelta':
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate,
                                             weight_decay=self.args.weight_decay)
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         weight_decay=self.args.weight_decay)
        elif self.args.optim == 'rprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            weight_decay=self.args.weight_decay)
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.args.optim))
        return optimizer

    def adjust_learning_rate(self, decay_rate=0.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def _train_epoch(self, train_batches, dataset, metric_save, patience, step_pbar):
        evaluate = True
        exit_tag = False
        check_point, batch_size = self.args.check_point, self.args.batch_size
        save_dir, save_prefix = self.args.model_dir, self.args.algo

        for bitx, batch in enumerate(train_batches):
            self.global_step += 1
            step_pbar.update(1)

            QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
            UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
            VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
            CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, :-2])
            TRUE_CLICKS = torch.from_numpy(np.array(batch['clicks'], dtype=np.float32)[:, 2:])
            input_ids = batch['candids'][0]["input_ids"]
            attention_mask = batch['candids'][0]["attention_mask"]
            token_type_ids = batch['candids'][0]["token_type_ids"]
            if use_cuda:
                QIDS, UIDS, VIDS, CLICKS, TRUE_CLICKS, input_ids, attention_mask, token_type_ids = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda(), TRUE_CLICKS.cuda(), input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(),

            self.model.train()
            self.optimizer.zero_grad()
            pred_logits = self.model(QIDS, UIDS, VIDS, CLICKS, input_ids, attention_mask, token_type_ids)
            loss = self.compute_loss(pred_logits, TRUE_CLICKS)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('train/loss', loss, self.global_step)

            if evaluate and self.global_step % self.eval_freq == 0:
                valid_batches = dataset.gen_mini_batches('valid', shuffle=False)
                valid_loss, valid_perplexity = self.evaluate(valid_batches, dataset)
                self.writer.add_scalar("valid/loss", valid_loss, self.global_step)
                self.writer.add_scalar("valid/perplexity", valid_perplexity, self.global_step)

                test_batches = dataset.gen_mini_batches('test',  shuffle=False)
                test_loss, test_perplexity = self.evaluate(test_batches, dataset)
                self.writer.add_scalar("test/loss", test_loss, self.global_step)
                self.writer.add_scalar("test/perplexity", test_perplexity, self.global_step)
                print("test/loss", test_loss, self.global_step)
                print("test/perplexity", test_perplexity, self.global_step)

                label_batches = dataset.gen_mini_batches('label', shuffle=False)
                ndcgs = self.ranking(label_batches, dataset)
                torch.cuda.empty_cache()
                for trunc_level in self.trunc_levels:
                    self.writer.add_scalar("rank/{}".format(trunc_level), ndcgs[trunc_level], self.global_step)
                    print("rank/{}".format(trunc_level), ndcgs[trunc_level], self.global_step)

                if valid_perplexity < metric_save:
                    metric_save = valid_perplexity
                    patience = 0
                else:
                    patience += 1
                if patience >= self.patience:
                    self.adjust_learning_rate(self.args.lr_decay)
                    self.learning_rate *= self.args.lr_decay
                    self.writer.add_scalar('train/lr', self.learning_rate, self.global_step)
                    metric_save = valid_perplexity
                    patience = 0
                    self.patience += 1
            if check_point > 0 and self.global_step % check_point == 0:
                self.save_model(save_dir, save_prefix)
            if self.global_step >= self.args.num_steps:
                exit_tag = True

        return exit_tag, metric_save, patience

    def train(self, dataset):
        patience, metric_save = 0, 1e10
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        self.writer.add_scalar('train/lr', self.args.learning_rate, self.global_step)
        while not exit_tag:
            train_batches = dataset.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            exit_tag, metric_save, patience = self._train_epoch(train_batches, dataset, metric_save, patience,
                                                                step_pbar)

    def evaluate(self, eval_batches, dataset):
        total_loss, total_num = 0., 0
        perplexity_num = 0
        perplexity_at_rank = torch.zeros(10, device=device, dtype=torch.float)  # 10 docs per query
        with torch.no_grad():
            for b_itx, batch in enumerate(eval_batches):

                QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
                UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
                VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
                CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, :-2])
                TRUE_CLICKS = torch.from_numpy(np.array(batch['clicks'], dtype=np.float32)[:, 2:])
                input_ids = batch['candids'][0]["input_ids"]
                attention_mask = batch['candids'][0]["attention_mask"]
                token_type_ids = batch['candids'][0]["token_type_ids"]
                if use_cuda:
                    QIDS, UIDS, VIDS, CLICKS, TRUE_CLICKS, input_ids, attention_mask, token_type_ids = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda(), TRUE_CLICKS.cuda(), input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(),

                self.model.eval()
                self.optimizer.zero_grad()
                pred_logits = self.model(QIDS, UIDS, VIDS, CLICKS, input_ids, attention_mask, token_type_ids)
                loss = self.compute_loss(pred_logits, TRUE_CLICKS)
                batch_perplexity_at_rank = self.compute_perplexity(pred_logits, TRUE_CLICKS)
                perplexity_at_rank = perplexity_at_rank + batch_perplexity_at_rank
                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])

            loss = 1.0 * total_loss / total_num
            perplexity = (2 ** (- perplexity_at_rank / total_num)).sum() / 10
        return loss, perplexity

    def ranking(self, label_batches, dataset):
        ndcgs, cnt_useless_session, cnt_usefull_session = {}, {}, {}
        for k in self.trunc_levels:
            ndcgs[k] = 0.0
            cnt_useless_session[k] = 0
            cnt_usefull_session[k] = 0
        with torch.no_grad():
            for b_idx, batch in enumerate(label_batches):
                QIDS = Variable(torch.from_numpy(np.array(batch['qids'], dtype=np.int64)))
                UIDS = Variable(torch.from_numpy(np.array(batch['uids'], dtype=np.int64)))
                VIDS = Variable(torch.from_numpy(np.array(batch['vids'], dtype=np.int64)))
                CLICKS = Variable(torch.from_numpy(np.array(batch['clicks'], dtype=np.int64))[:, :-2])
                TRUE_CLICKS = torch.from_numpy(np.array(batch['clicks'], dtype=np.float32)[:, 2:])
                input_ids = batch['candids'][0]["input_ids"]
                attention_mask = batch['candids'][0]["attention_mask"]
                token_type_ids = batch['candids'][0]["token_type_ids"]
                if use_cuda:
                    QIDS, UIDS, VIDS, CLICKS, TRUE_CLICKS, input_ids, attention_mask, token_type_ids = QIDS.cuda(), UIDS.cuda(), VIDS.cuda(), CLICKS.cuda(), TRUE_CLICKS.cuda(), input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(),

                self.model.eval()
                self.optimizer.zero_grad()
                pred_logits = self.model(QIDS, UIDS, VIDS, CLICKS, input_ids, attention_mask, token_type_ids)
                relevances_batches = pred_logits.data.cpu().numpy().tolist()
                true_relevances_batches = batch['relevances']

                for relevances, true_relevances in zip(relevances_batches, true_relevances_batches):
                    pred_rels = {}
                    for idx, relevance in enumerate(relevances):
                        pred_rels[idx] = relevance

                    for k in self.trunc_levels:
                        ideal_ranking_relevances = sorted(true_relevances, reverse=True)[:k]
                        ranking = sorted([idx for idx in pred_rels], key=lambda idx: pred_rels[idx], reverse=True)
                        ranking_relevances = [true_relevances[idx] for idx in ranking[:k]]
                        dcg = self.dcg(ranking_relevances)
                        idcg = self.dcg(ideal_ranking_relevances)
                        ndcg = dcg / idcg if idcg > 0 else 1.0
                        if idcg == 0:
                            cnt_useless_session[k] += 1
                        else:
                            ndcgs[k] += ndcg
                            cnt_usefull_session[k] += 1

            for k in self.trunc_levels:
                ndcgs[k] /= cnt_usefull_session[k]
        return ndcgs

    def dcg(self, ranking_relevances):
        """
        Compute the DCG for a given ranking_relevances
        """

        return sum([(2 ** relevance - 1) / math.log(rank + 2, 2) for rank, relevance in enumerate(ranking_relevances)])

    def save_model(self, model_dir, model_prefix):
        """
        Save the model into model_dir with model_prefix as the model indicator
        """
        torch.save(self.model.state_dict(),
                   os.path.join(model_dir, model_prefix + '_{}.model'.format(self.global_step)))
        torch.save(self.optimizer.state_dict(),
                   os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(self.global_step)))
        self.logger.info(
            'Model and optimizer saved in {}, with prefix {} and global step {}.'.format(model_dir, model_prefix,
                                                                                         self.global_step))

    def load_model(self, model_dir, model_prefix, global_step):
        """
        Reload the model into model_dir from model_prefix as the model indicator
        """
        optimizer_path = os.path.join(model_dir, model_prefix + '_{}.optimizer'.format(global_step))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        self.logger.info(
            'Optimizer reloaded from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix,
                                                                                    global_step))
        model_path = os.path.join(model_dir, model_prefix + '_{}.model'.format(global_step))
        if use_cuda:
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.logger.info(
            'Model restored from {}, with prefix {} and global step {}.'.format(model_dir, model_prefix, global_step))
