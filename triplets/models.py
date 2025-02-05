import numpy as np
import torch
import pandas as pd
from torch.nn import PairwiseDistance, CosineSimilarity

class Ratings(torch.utils.data.Dataset):

    def __init__(self, items=None, raters=None, triplets=None):
        # For one-hot encoding, can be extended to 'proposal function'
        self.items = items.index.values
        self.num_items, self.dim_items = items.shape
        self.E_item = torch.nn.Embedding(self.num_items, self.dim_items)
        self.E_item.weight.data = torch.Tensor(items.values)
        self.E_item.weight.requires_grad = False
        if raters is not None:
            self.raters = raters.index.values
            self.num_raters, self.dim_raters = raters.shape
            self.E_rater = torch.nn.Embedding(self.num_raters, self.dim_raters)
            self.E_rater.weight.data = torch.Tensor(raters.values)
            self.E_rater.weight.requires_grad = False
        self.E_position = torch.nn.Embedding(4, 3)
        self.E_position.weight.data.fill_(0.)
        self.E_position.weight.data[:3,:] = torch.eye(3)
        self.triplets = []
        self.cost = 0.
        if triplets is not None:
            self.add_triplets(triplets)

    def item2index(self, items):
        return np.array([np.where(self.items==item)[0][0] for item in items])

    def index2item(self, index):
        return self.items[np.array(index)]

    def rater2index(self, raters):
        return np.array([np.where(self.raters==rater)[0][0] for rater in raters])

    def index2rater(self, index):
        return self.raters[np.array(index)]

    def embed(self, items, rater, last_selected):
        return (self.E_item(torch.LongTensor(items[None])),
                self.E_rater(torch.LongTensor(rater[None])),
                self.E_position(torch.LongTensor(last_selected[None])))

    def add_triplets(self, triplets):
        # Input comes as a pandas dataframe
        for index in triplets.index:
            row = triplets.loc[index]
            stimuli = self.item2index(row.loc[['stim_0','stim_1','stim_2']].values)
            rater = self.rater2index([row['rater']])[0]
            self.triplets.append((stimuli, rater, row['last_selected'], row['selected']))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        items, rater, last_selected, y = self.triplets[idx]
        x_items, x_rater, x_last_selected = self.embed(items, rater, last_selected)
        return (x_items, x_rater, x_last_selected), y

class CollateTrials:

    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        """
        batch: list of tuples (input wav, phoneme labels, word labels)

        Returns a minibatch of wavs and labels as Tensors.
        """
        x_i = []; x_r = []; x_p = []; y = []; lengths = [];
        batch_size = len(batch)
        for index in range(batch_size):
            (x_items_, x_rater_, x_last_selected_), y_ = batch[index]
            x_i.append(x_items_.float()[0])
            x_r.append(x_rater_.float()[0])
            x_p.append(x_last_selected_.float()[0])
            y.append(y_)
            lengths.append(len(x_items_[0]))
        x_i = torch.nn.utils.rnn.pad_sequence(x_i,batch_first=True)
        x_r = torch.vstack(x_r)
        x_p = torch.vstack(x_p)
        y = torch.LongTensor(y)
        lengths = torch.LongTensor(lengths)
        return ((x_i.to(self.device), x_r.to(self.device), x_p.to(self.device)),
                y.to(self.device), lengths.to(self.device))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class RaterModelTorch(torch.nn.Module):

    def __init__(self, dim_items):
        super(RaterModelTorch, self).__init__()
        self.dim_items = dim_items

    def update(self, dataset, lr=.01, min_lr=1e-5, max_iter=200, patience=5, verbose=False, validset=None):
        if not np.any([param.requires_grad for param in self.parameters()]):
            return []
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=patience)
        self.train()
        old_loss = 1e3
        best_loss = 1e3
        loss_diff = 1.
        i_iter = 0
        losses = []
        num_batches = len(dataset.loader)
        if verbose:
            print('Starting')
        while (i_iter<max_iter) and (lr>min_lr):
            # print(i_iter)
            # print(self.linear_sequence.weight[0])
            i_iter += 1
            new_loss = 0.
            for batch in dataset.loader:
                x, y, lengths = batch
                yhat = self.forward(x, lengths)
                optimizer.zero_grad()
                loss = self.loss(yhat,y)[0]
                loss.backward()
                optimizer.step()
                new_loss += loss.item()/num_batches
            lr = get_lr(optimizer)
            if validset is not None:
                new_loss, _ = self.test(validset)
            if new_loss<best_loss:
                best_loss = new_loss
                torch.save(self.state_dict(), self.state_dict_path)
            if verbose:
                print('Iteration {}, lr {}, loss {}'.format(i_iter,lr,new_loss))
                # print('sequence: {}'.format(self.linear_sequence.weight.data))
            scheduler.step(new_loss)
            loss_diff = np.abs(new_loss-old_loss)
            old_loss = new_loss
            losses.append(new_loss)
        if verbose:
            print('Iteration {}, lr {}, loss {}'.format(i_iter,lr,best_loss))
        self.load_state_dict(torch.load(self.state_dict_path))
        self.eval()
        return losses

    def test(self, testset):
        valid_loss = 0.
        corrects = 0
        num_batches = 0
        for batch in testset.loader:
            x, y, lengths = batch
            bsz = y.shape[0]
            yhat = self.forward(x)
            corrects += sum(y==yhat.data.argmax(1)).item()
            loss = self.loss(yhat,y)[1]
            valid_loss += loss.item()*bsz
            num_batches += bsz
        return valid_loss/num_batches, corrects/len(testset)

    def test_individual(self, testset):
        valid_loss = 0.
        corrects = 0
        num_batches = 0
        bits = []
        corrects = []
        for batch in testset.loader:
            x, y, lengths = batch
            yhat = self.forward(x)
            corrects += (y==yhat.data.argmax(1)).tolist()
            bits += self.loss_bits(yhat,y).tolist()
        rater_indices = [triplet[1] for triplet in testset.triplets]
        return bits, corrects, testset.index2rater(rater_indices).tolist()

class RaterModelTriplet(RaterModelTorch):

    def __init__(
        self, dim_items, dim_raters, num_dims, lambda_item, lambda_rater, 
        distance, force_positive, fit_items, fit_raters, fit_sequence, 
        reg_type, state_dict_path, device='cpu', rater_mask=None
    ):
        super(RaterModelTriplet, self).__init__(dim_items)
        self.num_dims = num_dims
        self.lambda_item = lambda_item
        if lambda_rater is None:
            self.lambda_rater = lambda_item
        else:
            self.lambda_rater = lambda_rater
        self.linear_item = torch.nn.Linear(dim_items, num_dims, bias=False)
        self.linear_rater = torch.nn.Linear(dim_raters, num_dims, bias=False)
        if force_positive=='softplus':
            self.linear_rater.weight.data.fill_(0.)
        else:
            self.linear_item.weight.data.fill_(0.)
        self.linear_sequence = torch.nn.Linear(dim_raters, 1, bias=False)
        self.force_positive = force_positive
        self.distance = distance

        if fit_items:
            self.linear_item.weight.requires_grad = True
        else:
            self.linear_item.weight.requires_grad = False

        if fit_raters:
            self.linear_rater.weight.requires_grad = True
        else:
            self.linear_rater.weight.requires_grad = False

        if fit_sequence:
            self.linear_sequence.weight.data.fill_(0.)
            self.linear_sequence.weight.requires_grad = True
        else:
            self.linear_sequence.weight.data.fill_(0.)
            self.linear_sequence.weight.requires_grad = False            

        self.reg_type = reg_type
        if device=='cuda':
            self.cuda()
        self.state_dict_path = state_dict_path
        if rater_mask is None:
            self.rater_mask = torch.arange(dim_raters)
        else:
            self.rater_mask = rater_mask

    def forward(self, x, lengths=None):
        x_item, x_rater, x_last_selected = x
        x_item_embed = self.linear_item(x_item)
        x_rater_embed = self.linear_rater(x_rater)
        x_sequence_embed = self.linear_sequence(x_rater)
        if self.force_positive=='sigmoid':
            x_item_embed = torch.sigmoid(x_item_embed)
            x_item_embed = x_item_embed * torch.nn.functional.softplus(x_rater_embed[:,None,:])
        elif self.force_positive=='softplus':
            x_item_embed = torch.nn.functional.softplus(x_item_embed)
            x_item_embed = x_item_embed * torch.sigmoid(x_rater_embed[:,None,:])
        bsz = x_item.shape[0]
        embed_left = x_item_embed[:,[1,0,0],:].reshape(bsz*3,-1)
        embed_right = x_item_embed[:,[2,2,1],:].reshape(bsz*3,-1)
        if self.distance=='euclidean':
            proximity = 1-PairwiseDistance()(embed_left, embed_right).reshape(bsz,-1)
        elif self.distance=='cosine':
            proximity = CosineSimilarity()(embed_left, embed_right).reshape(bsz,-1)
        else: # dot product
            proximity = (embed_left*embed_right).sum(1).reshape(bsz,-1)
        proximity_out = torch.exp(proximity + x_last_selected.data * x_sequence_embed)
        return proximity_out

    def loss(self, proximity, y):
        bsz = y.shape[0]
        numerators = proximity[torch.arange(bsz),y]
        denominators = proximity.sum(1)
        crossentropy_loss = -torch.log(numerators/denominators).sum()/bsz
        params_item = torch.cat([x.view(-1) for x in self.linear_item.parameters()])
        # params_rater = self.linear_rater.weight[:,self.rater_mask].view(-1)
        params_rater = torch.cat([self.linear_sequence.weight[:,self.rater_mask].view(-1),
                                     self.linear_rater.weight[:,self.rater_mask].view(-1)])
        norm_item = torch.norm(params_item, self.reg_type)
        # print('{} - {}'.format(norm_item, norm_item * (len(params_item)**(-1/self.reg_type))))
        # print(self.linear_item.weight[0])
        norm_rater = torch.norm(params_rater, self.reg_type)
        norm_item = norm_item * (len(params_item)**(-1/self.reg_type))
        norm_rater = norm_rater * (len(params_rater)**(-1/self.reg_type))
        regularization = self.lambda_item*norm_item + self.lambda_rater*norm_rater
        # print('{}: {} - {}'.format(self.lambda_rater, crossentropy_loss, regularization))
        return crossentropy_loss + regularization, crossentropy_loss
    

    def loss_bits(self, proximity, y):
        bsz = y.shape[0]
        numerators = proximity[torch.arange(bsz),y]
        denominators = proximity.sum(1)
        return -torch.log2(numerators/denominators)