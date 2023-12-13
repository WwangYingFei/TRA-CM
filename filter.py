import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
from torch.autograd import Variable
import logging
import torch.nn.functional as F
from transformers import BertModel

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
INF = 1e30
# device = torch.device('cpu')

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):

    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class Embedding(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size):
        super(Embedding, self).__init__()
        self.args = args
        self.logger = logging.getLogger("")
        self.query_size = query_size
        self.doc_size = doc_size
        self.vtype_size = vtype_size
        self.dataset = args.dataset
        self.data_dir = os.path.join('data', self.dataset)

        self.qid_embedding = nn.Embedding(query_size, self.args.embed_size)
        self.uid_embedding = nn.Embedding(doc_size, self.args.embed_size)
        self.click_embedding = nn.Embedding(2, self.args.click_embed_size)
        self.vid_embedding = nn.Embedding(vtype_size, self.args.vtype_embed_size)
        self.pos_embedding = nn.Embedding(10, self.args.pos_embed_size)


    def forward(self, qids, uids, vids, clicks):

        batch_size = clicks.shape[0]
        seq_len = clicks.shape[1]
        qid_embedding = self.qid_embedding(qids)
        uid_embedding = self.uid_embedding(uids)
        click_embedding = self.click_embedding(clicks)  # [batch_size, seq_len, click_embed_size]
        vid_embedding = self.vid_embedding(vids)  # [batch_size, seq_len, vtype_embed_size]
        pos_embedding = self.pos_embedding.weight.unsqueeze(dim=0).repeat(batch_size, seq_len // 10, 1)

        return qid_embedding, uid_embedding, click_embedding, vid_embedding, pos_embedding


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        # 一个极小的值，避免分母为0
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        return self.weight * x + self.bias

class FilterLayer(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.seq_len//2 + 1, args.embed_size, 2, dtype=torch.float32) * 0.02)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.LayerNorm = LayerNorm(args.embed_size, eps=1e-12)
        self.embedding = Embedding(args, query_size, doc_size, vtype_size)

    def forward(self, qids, uids, vids, clicks):
        query_embedding, url_embedding, vtype_embedding, click_embedding, rank_embedding = self.embedding(qids, uids, vids, clicks)
        batch, seq_len, embed = query_embedding.shape
        weight = torch.view_as_complex(self.complex_weight)
        qid = torch.fft.rfft(query_embedding, n=seq_len, dim=1, norm='ortho')
        qid = qid * weight
        qid_emb_fft = torch.fft.irfft(qid, n=seq_len, dim=1, norm='ortho')
        qid_states = self.dropout(qid_emb_fft)
        qid_states = self.LayerNorm(qid_states + query_embedding)

        uid = torch.fft.rfft(url_embedding, n=seq_len, dim=1, norm='ortho')
        uid = uid * weight
        uid_emb_fft = torch.fft.irfft(uid, n=seq_len, dim=1, norm='ortho')
        uid_states = self.dropout(uid_emb_fft)
        uid_states = self.LayerNorm(uid_states + url_embedding)

        vid = torch.fft.rfft(vtype_embedding, n=seq_len, dim=1, norm='ortho')
        vid = vid * weight
        vid_emb_fft = torch.fft.irfft(vid, n=seq_len, dim=1, norm='ortho')
        vid_states = self.dropout(vid_emb_fft)
        vid_states = self.LayerNorm(vid_states + vtype_embedding)

        click = torch.fft.rfft(click_embedding, n=seq_len, dim=1, norm='ortho')
        click = click * weight
        click_emb_fft = torch.fft.irfft(click, n=seq_len, dim=1, norm='ortho')
        click_states = self.dropout(click_emb_fft)
        click_states = self.LayerNorm(click_states + click_embedding)

        pos = torch.fft.rfft(rank_embedding, n=seq_len, dim=1, norm='ortho')
        pos = pos * weight
        pos_emb_fft = torch.fft.irfft(pos, n=seq_len, dim=1, norm='ortho')
        pos_states = self.dropout(pos_emb_fft)
        pos_states = self.LayerNorm(pos_states + rank_embedding)

        return qid_states, uid_states, vid_states, click_states, pos_states

class Intermediate(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(Intermediate, self).__init__()
        self.linear_1 = nn.Linear(args.embed_size, args.embed_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.linear_2 = nn.Linear(4 * args.embed_size, args.embed_size)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.LayerNorm = LayerNorm(args.embed_size, eps=1e-12)
        self.filterlayer = FilterLayer(args, query_size, doc_size, vtype_size, dataset)

    def forward(self, qids, uids, vids, clicks):
        qid_states, uid_states, vid_states, click_states, pos_states = self.filterlayer(qids, uids, vids, clicks)
        qid_hidden_states = self.linear_1(qid_states)
        qid_hidden_states = self.intermediate_act_fn(qid_hidden_states)
        qid_hidden_states = self.linear_2(qid_hidden_states)
        qid_hidden_states = self.dropout(qid_hidden_states)
        qid_hidden_states = self.LayerNorm(qid_hidden_states + qid_states)

        uid_hidden_states = self.linear_1(uid_states)
        uid_hidden_states = self.intermediate_act_fn(uid_hidden_states)
        uid_hidden_states = self.linear_2(uid_hidden_states)
        uid_hidden_states = self.dropout(uid_hidden_states)
        uid_hidden_states = self.LayerNorm(uid_hidden_states + uid_states)

        vid_hidden_states = self.linear_1(vid_states)
        vid_hidden_states = self.intermediate_act_fn(vid_hidden_states)
        vid_hidden_states = self.linear_2(vid_hidden_states)
        vid_hidden_states = self.dropout(vid_hidden_states)
        vid_hidden_states = self.LayerNorm(vid_hidden_states + vid_states)

        click_hidden_states = self.linear_1(click_states)
        click_hidden_states = self.intermediate_act_fn(click_hidden_states)
        click_hidden_states = self.linear_2(click_hidden_states)
        click_hidden_states = self.dropout(click_hidden_states)
        click_hidden_states = self.LayerNorm(click_hidden_states + click_states)

        pos_hidden_states = self.linear_1(pos_states)
        pos_hidden_states = self.intermediate_act_fn(pos_hidden_states)
        pos_hidden_states = self.linear_2(pos_hidden_states)
        pos_hidden_states = self.dropout(pos_hidden_states)
        pos_hidden_states = self.LayerNorm(pos_hidden_states + pos_states)

        return qid_hidden_states, uid_hidden_states, vid_hidden_states, click_hidden_states, pos_hidden_states

class ExamPredictor(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(ExamPredictor, self).__init__()
        self.args = args
        self.exam_gru = nn.GRU(self.args.pos_embed_size + self.args.vtype_embed_size + self.args.click_embed_size,
                               self.args.hidden_size, batch_first=True)
        self.embedding = Embedding(args, query_size, doc_size, vtype_size)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_attention = args.use_attention
        self.linear = nn.Linear(self.args.hidden_size, 1)
        self.intermediate = Intermediate(args, query_size, doc_size, vtype_size, dataset)
        self.exam_out_dim = self.args.hidden_size
        self.exam_output_linear = nn.Linear(self.exam_out_dim, 1)
        self.dropout = nn.Dropout(p=self.args.dropout_rate)
        self.activation = nn.Sigmoid()

    def forward(self, qids, uids, vids, clicks):
        qid_hidden_states, uid_hidden_states, vid_hidden_states, click_hidden_states, pos_hidden_states = self.intermediate(qids, uids, vids, clicks)
        batch_size = qid_hidden_states.shape[0]
        seq_len = qid_hidden_states.shape[1]
        exam_input = torch.cat((vid_hidden_states, click_hidden_states, pos_hidden_states), dim=2)
        exam_state = Variable(torch.zeros(1, batch_size, self.args.hidden_size, device=device))
        exam_outputs, exam_state = self.exam_gru(exam_input, exam_state)
        exam_outputs = self.dropout(exam_outputs)
        exams = self.exam_output_linear(exam_outputs)
        exams = self.activation(exams).view(batch_size, seq_len) # [1,10]

        return exams

class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.logger = logging.getLogger("Transformer_Click_Model")
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.bert = BertModel.from_pretrained(".\\bert-base-chinese")
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = nn.Linear(768, 1)
        self.drop = nn.Dropout(0.2)

    # input包含tokenize
    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = outputs.last_hidden_state  # [10,128,768]
        output_tensor = torch.stack([last_hidden_states[i][0] for i in range(len(last_hidden_states))])  # [10,768]
        output = self.drop(output_tensor)
        output = self.linear(output)  # [10,1]
        topic_rels = output.unsqueeze(0) #[1,10,1]
        if use_cuda:
            topic_rels = topic_rels .cuda()

        return topic_rels

class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.logger = logging.getLogger("Transformer_Click_Model")
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = nn.Linear(768, 1)
        self.drop = nn.Dropout(0.2)

    # input包含tokenize
    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_states = outputs.last_hidden_state  # [10,128,768]
        output_tensor = torch.stack([last_hidden_states[i][0] for i in range(len(last_hidden_states))])  # [10,768]
        output = self.drop(output_tensor)
        output = self.linear(output)  # [10,1]
        topic_rels = output.unsqueeze(0) #[1,10,1]
        if use_cuda:
            topic_rels = topic_rels .cuda()

        return topic_rels

class Transformer(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(Transformer, self).__init__()
        self.args = args
        self.logger = logging.getLogger("Transformer_Click_Model")
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.dropout_rate = args.dropout_rate
        self.query_size = query_size
        self.doc_size = doc_size
        self.embedding = Embedding(args, query_size, doc_size, vtype_size)
        self.intermediate = Intermediate(args, query_size, doc_size, vtype_size, dataset)

        # Network
        self.transformerLayer = nn.TransformerEncoderLayer(128, 8, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.transformerLayer, num_layers=1)
        self.transformer_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.LeakyReLU(),
        )
        self.dropout = nn.Dropout(p=self.args.dropout_rate)
        self.activation = nn.Sigmoid()

    def forward(self, qids, uids, vids, clicks):
        qid_hidden_states, uid_hidden_states, vid_hidden_states, click_hidden_states, pos_hidden_states = self.intermediate(
                qids, uids, vids, clicks)

        uid = self.transformer_encoder(uid_hidden_states)
        vid = self.transformer_encoder(vid_hidden_states)
        click = self.transformer_encoder(click_hidden_states)
        pos = self.transformer_encoder(pos_hidden_states)

        transformer_outputs = torch.cat((uid, vid, click, pos), dim=2)
        outputs = self.dropout(transformer_outputs)
        user_rels = self.transformer_linear(outputs)

        return user_rels

class RelevanceCombination(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(RelevanceCombination, self).__init__()
        self.args = args
        self.transformer = Transformer(self.args, query_size, doc_size, vtype_size, dataset)
        self.bert = Bert(self.args)
        self.activation = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.5)
        self.beta.data.fill_(0.5)

    def forward(self, qids, uids, vids, clicks, input_ids, attention_mask, token_type_ids):
        batch_size = qids.shape[0]
        seq_len = qids.shape[1]
        user_rels = self.transformer(qids, uids, vids, clicks)
        topic_rels = self.bert(input_ids, attention_mask, token_type_ids)
        rels = torch.add(torch.mul(self.alpha, user_rels), torch.mul(self.beta, topic_rels))
        rels = self.activation(rels).view(batch_size, seq_len)  # [1,10]

        return rels





