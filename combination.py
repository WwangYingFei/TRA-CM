# encoding:utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
from filter import ExamPredictor, RelevanceCombination
# from filter import ExamPredictor, Transformer

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
INF = 1e30
MINF = -(1e30)
# device = torch.device('cpu')

class ClickModel(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(ClickModel, self).__init__()
        self.args = args
        self.dataset = dataset
        self.dropout = nn.Dropout(p=self.args.dropout_rate)
        self.sigmoid = nn.Sigmoid()

        # Examination Predictor
        self.exam_predictor = ExamPredictor(self.args, query_size, doc_size, vtype_size, dataset)

        # Relevance Estimator
        self.relevance_estimator = RelevanceCombination(args, query_size, doc_size, vtype_size, dataset)

        # Combination Layer
        if self.args.combine == 'exp_mul' or self.args.combine == 'exp_sigmoid_log':
            self.lamda = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.mu = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.lamda.data.fill_(1.0)
            self.mu.data.fill_(1.0)
        elif self.args.combine == 'linear':
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.mu = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.alpha.data.fill_(0.5)
            self.beta.data.fill_(0.5)
            self.mu.data.fill_(0.5)
        elif self.args.combine == 'nonlinear':
            self.w11 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w12 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w21 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w22 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w31 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w32 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w11.data.fill_(0.5)
            self.w12.data.fill_(0.5)
            self.w21.data.fill_(0.5)
            self.w22.data.fill_(0.5)
            self.w31.data.fill_(0.5)
            self.w32.data.fill_(0.5)

    def combine(self, exams, rels):
        '''
        Combine examination and relevance to get the click probability
        '''
        combine = self.args.combine
        if combine == 'linear_add':
            clicks = torch.add(torch.mul(rels, self.alpha),torch.mul(exams,(1 - self.alpha)))
        if combine == 'mul':
            clicks = torch.mul(rels, exams)
        # torch.pow(input, exponent)
        elif combine == 'exp_mul':
            clicks = torch.mul(torch.pow(rels, self.lamda), torch.pow(exams, self.mu))
        elif combine == 'linear':
            clicks = torch.add(torch.mul(exams, self.alpha), torch.mul(rels, self.beta))
        elif combine == 'nonlinear':  # 2-layer
            out1 = self.sigmoid(torch.add(torch.mul(rels, self.w11), torch.mul(exams, self.w12)))
            out2 = self.sigmoid(torch.add(torch.mul(rels, self.w21), torch.mul(exams, self.w22)))
            clicks = self.sigmoid(torch.add(torch.mul(out1, self.w31), torch.mul(out2, self.w32)))
        elif combine == 'sigmoid_log':
            clicks = 4 * torch.div(torch.mul(rels, exams),
                                   torch.mul(torch.add(rels, 1), torch.add(exams, 1)))
        else:
            raise NotImplementedError('Unsupported combination type: {}'.format(combine))
        return clicks

    def forward(self, qids, uids, vids, clicks, input_ids, attention_mask, token_type_ids):
        batch_size = len(qids)
        seq_len = len(qids[0])

        # Examination predition process
        exams = self.exam_predictor(qids, uids, vids, clicks)

        # Relevance estimation process
        rels = self.relevance_estimator(qids, uids, vids, clicks, input_ids, attention_mask, token_type_ids)
        # rels = self.relevance_estimator(qids, uids, vids, clicks)

        # Combination Layer
        pred_logits = self.combine(exams, rels)
        pred_logits = pred_logits.view(batch_size, seq_len)

        return pred_logits