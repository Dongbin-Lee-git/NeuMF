import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
import pandas as pd
torch.manual_seed(0)
from tqdm import tqdm
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True

class Train_Dataset(Dataset):
    def __init__(self, ratings, all_jobIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_jobIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, train, all_jobIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(train['userID'], train['jobID']))

        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_jobIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_jobIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)

class GMF(pl.LightningModule):
    def __init__(self, num_users, num_items, ratings, all_jobIds):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=32)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_jobIds = all_jobIds

    def forward(self, user_input, item_input):
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.mul(user_embedded, item_embedded)

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        # print('\nLoss = '+ loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(Train_Dataset(self.ratings, self.all_jobIds),
                          batch_size=256, num_workers=1)

class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
    """

    def __init__(self, num_users, num_items, ratings, all_jobIds, pre_train = False , user_tag_emb = None, job_tag_emb = None):
        super().__init__()
        self.pre_train = pre_train
        self.user_tag_emb = user_tag_emb
        self.job_tag_emb = job_tag_emb

        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=128)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=128)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)

        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_jobIds = all_jobIds

        if pre_train:
            self._init_weight_()

    def _init_weight_(self):
        self.user_embedding.from_pretrained(self.user_tag_emb, freeze=False)
        self.item_embedding.from_pretrained(self.job_tag_emb, freeze=False)

    def forward(self, user_input, item_input):
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        vector = nn.ReLU()(self.fc3(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        #print('\nLoss = '+ loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(Train_Dataset(self.ratings, self.all_jobIds),
                          batch_size=256, num_workers=1)

class NeuMF(pl.LightningModule):

    def __init__(self, user_num, item_num, ratings, all_jobIds, GMF_model=None,
                 user_tag_emb=None, job_tag_emb=None, MLP_model=None, pre_train=False):
        super().__init__()
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        self.pre_train = pre_train
        self.user_tag_emb = user_tag_emb
        self.job_tag_emb = job_tag_emb

        self.embed_user_GMF = nn.Embedding(user_num, 32)
        self.embed_item_GMF = nn.Embedding(item_num, 32)
        self.embed_user_MLP = nn.Embedding(user_num, 128)
        self.embed_item_MLP = nn.Embedding(item_num, 128)

        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)

        self.predict_layer = nn.Linear(64, 1)
        if pre_train:
            self._init_weight_()
        self.ratings = ratings
        self.all_jobIds = all_jobIds

    def _init_weight_(self):
        self.embed_user_MLP.from_pretrained(self.user_tag_emb, freeze=False)
        self.embed_item_MLP.from_pretrained(self.job_tag_emb, freeze=False)

        self.embed_user_GMF.weight.data.copy_(
            self.GMF_model['state_dict']['user_embedding.weight'])
        self.embed_item_GMF.weight.data.copy_(
            self.GMF_model['state_dict']['user_embedding.weight'])
        self.embed_user_MLP.weight.data.copy_(
            self.MLP_model['state_dict']['user_embedding.weight'])
        self.embed_item_MLP.weight.data.copy_(
            self.MLP_model['state_dict']['user_embedding.weight'])

        self.fc1.weight.data.copy_(self.MLP_model['state_dict']['fc1.weight'])
        self.fc1.bias.data.copy_(self.MLP_model['state_dict']['fc1.bias'])

        self.fc2.weight.data.copy_(self.MLP_model['state_dict']['fc2.weight'])
        self.fc2.bias.data.copy_(self.MLP_model['state_dict']['fc2.bias'])

        self.fc3.weight.data.copy_(self.MLP_model['state_dict']['fc3.weight'])
        self.fc3.bias.data.copy_(self.MLP_model['state_dict']['fc3.bias'])

        # predict layers
        predict_weight = torch.cat([
            self.GMF_model['state_dict']['output.weight'],
            self.MLP_model['state_dict']['output.weight']], dim=1)

        predict_bias = self.GMF_model['state_dict']['output.bias'] + \
                      self.MLP_model['state_dict']['output.bias']

        self.predict_layer.weight.data.copy_(0.5 * predict_weight)
        self.predict_layer.bias.data.copy_(0.5 * predict_bias)


    def forward(self, user, item):
        # Pass through embedding layers
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)

        vector = nn.ReLU()(self.fc1(interaction))
        vector = nn.ReLU()(self.fc2(vector))
        output_MLP = nn.ReLU()(self.fc3(vector))

        # Concat the two embedding layers
        concat = torch.cat((output_GMF, output_MLP), -1)

        # Output layer
        pred = nn.Sigmoid()(self.predict_layer(concat))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(Train_Dataset(self.ratings, self.all_jobIds),
                          batch_size=256, num_workers=1)

def run_pretrain():
    torch.multiprocessing.freeze_support()
    #df_tags = pd.read_csv('tags.csv')

    #### 데이터셋 불러오기 ####
    df_job_tags = pd.read_csv('job_tags.csv')
    df_job_companies = pd.read_csv('job_companies.csv')
    df_train = pd.read_csv('train.csv')
    df_user_tags = pd.read_csv('user_tags.csv')

    ###테스트셋####
    df_test = pd.read_csv('test_job.csv')

    # mapping
    all_jobids = df_job_companies['jobID'].unique() #733
    all_userids = df_user_tags['userID'].unique() #196

    userid2idx = {o:i for i,o in enumerate(all_userids)}
    jobid2idx = {o:i for i,o in enumerate(all_jobids)}

    df_train['userID'] = df_train['userID'].apply(lambda x: userid2idx[x])
    df_train['jobID'] = df_train['jobID'].apply(lambda x: jobid2idx[x])

    df_test['userID'] = df_test['userID'].apply(lambda x: userid2idx[x])
    df_test['jobID'] = df_test['jobID'].apply(lambda x: jobid2idx[x])


    # Placeholders that will hold the training data
    users, items, labels = [], [], []

    # This is the set of items that each user has interaction with
    user_item_set = set(zip(df_train['userID'], df_train['jobID']))

    # 4:1 ratio of negative to positive samples
    num_negatives = 4

    for (u, i) in tqdm(user_item_set):
        users.append(u)
        items.append(i)
        labels.append(1) # items that the user has interacted with are positive
        for _ in range(num_negatives):
            # randomly select an item
            negative_item = np.random.choice(all_jobids)
            # check that the user has not interacted with this item
            while (u, negative_item) in user_item_set:
                negative_item = np.random.choice(all_jobids)
            users.append(u)
            items.append(negative_item)
            labels.append(0) # items not interacted with are negative

    num_users = len(df_train['userID'])
    num_items = len(df_train['jobID'])
    all_jobids = df_train['jobID'].unique()

    user_tag_emb = torch.tensor(np.load('save_emb/user_tag_emb_np.npy'))
    job_tag_emb = torch.tensor(np.load('save_emb/job_tag_emb_np.npy'))

    model_ncf = NCF(num_users, num_items, df_train, all_jobids, pre_train=True,
                    user_tag_emb=user_tag_emb, job_tag_emb=job_tag_emb)

    trainer_ncf = pl.Trainer(max_epochs=10, gpus=1, reload_dataloaders_every_epoch=True,
                         progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)

    trainer_ncf.fit(model_ncf)
    trainer_ncf.save_checkpoint("save_model/ncf_epo10_bat256_mul3")


    model_gmf = GMF(num_users, num_items, df_train, all_jobids)

    trainer_gmf = pl.Trainer(max_epochs=10, gpus=1, reload_dataloaders_every_epoch=True,
                             progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)

    trainer_gmf.fit(model_gmf)
    trainer_gmf.save_checkpoint("save_model/gmf_epo10_bat256")

def run_finetune():
    torch.multiprocessing.freeze_support()
    #df_tags = pd.read_csv('tags.csv')

    #### 데이터셋 불러오기 ####
    df_job_tags = pd.read_csv('job_tags.csv')
    df_job_companies = pd.read_csv('job_companies.csv')
    df_train = pd.read_csv('train.csv')
    df_user_tags = pd.read_csv('user_tags.csv')

    ###테스트셋####
    df_test = pd.read_csv('test_job.csv')

    # mapping
    all_jobids = df_job_companies['jobID'].unique() #733
    all_userids = df_user_tags['userID'].unique() #196

    userid2idx = {o:i for i,o in enumerate(all_userids)}
    jobid2idx = {o:i for i,o in enumerate(all_jobids)}

    df_train['userID'] = df_train['userID'].apply(lambda x: userid2idx[x])
    df_train['jobID'] = df_train['jobID'].apply(lambda x: jobid2idx[x])

    df_test['userID'] = df_test['userID'].apply(lambda x: userid2idx[x])
    df_test['jobID'] = df_test['jobID'].apply(lambda x: jobid2idx[x])


    # Placeholders that will hold the training data
    users, items, labels = [], [], []

    # This is the set of items that each user has interaction with
    user_item_set = set(zip(df_train['userID'], df_train['jobID']))

    # 4:1 ratio of negative to positive samples
    num_negatives = 4

    for (u, i) in tqdm(user_item_set):
        users.append(u)
        items.append(i)
        labels.append(1) # items that the user has interacted with are positive
        for _ in range(num_negatives):
            # randomly select an item
            negative_item = np.random.choice(all_jobids)
            # check that the user has not interacted with this item
            while (u, negative_item) in user_item_set:
                negative_item = np.random.choice(all_jobids)
            users.append(u)
            items.append(negative_item)
            labels.append(0) # items not interacted with are negative

    num_users = len(df_train['userID'])
    num_items = len(df_train['jobID'])
    all_jobids = df_train['jobID'].unique()

    GMF_model = torch.load('save_model/gmf_epo10_bat256')
    MLP_model = torch.load('save_model/ncf_epo10_bat256_mul3')

    user_tag_emb = torch.tensor(np.load('save_emb/user_tag_emb_np.npy'))
    job_tag_emb = torch.tensor(np.load('save_emb/job_tag_emb_np.npy'))

    model_neumf = NeuMF(num_users, num_items, df_train, all_jobids, user_tag_emb=user_tag_emb, job_tag_emb=job_tag_emb,
                        GMF_model=GMF_model, MLP_model=MLP_model, pre_train=True)
    trainer_neumf = pl.Trainer(max_epochs=20, gpus=1, reload_dataloaders_every_epoch=True,
                             progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)

    trainer_neumf.fit(model_neumf)
    trainer_neumf.save_checkpoint("save_model/neumf_epo10_bat256_neumf_pre_train")


    #prediction
    pred = []
    test_user_item_set = set(zip(df_test['userID'], df_test['jobID']))

    for (u, i) in tqdm(test_user_item_set):
        pred.append(model_neumf(torch.tensor(u), torch.tensor(i)).detach().numpy().tolist())
    return pred

def pre_train_emb_tag():
    df_job_tags = pd.read_csv('job_tags.csv')
    df_job_companies = pd.read_csv('job_companies.csv')
    df_train = pd.read_csv('train.csv')
    df_user_tags = pd.read_csv('user_tags.csv')
    df_tags = pd.read_csv('tags.csv')

    # mapping
    all_jobids = df_job_companies['jobID'].unique() #733
    all_userids = df_user_tags['userID'].unique() #196
    all_tid = df_tags.tagID.unique() #887

    userid2idx = {o:i for i,o in enumerate(all_userids)}
    jobid2idx = {o:i for i,o in enumerate(all_jobids)}
    tagid2idx = {o: i for i, o in enumerate(all_tid)}

    df_train['userID'] = df_train['userID'].apply(lambda x: userid2idx[x])
    df_train['jobID'] = df_train['jobID'].apply(lambda x: jobid2idx[x])

    df_user_tags['userID'] = df_user_tags['userID'].apply(lambda x: userid2idx[x])
    df_user_tags['tagID'] = df_user_tags['tagID'].apply(lambda x: tagid2idx[x])

    df_job_tags['jobID'] = df_job_tags['jobID'].apply(lambda x: jobid2idx[x])
    df_job_tags['tagID'] = df_job_tags['tagID'].apply(lambda x: tagid2idx[x])

    user_interacted_tags = df_user_tags.groupby('userID')['tagID'].apply(list).to_dict()
    job_interacted_tags = df_job_tags.groupby('jobID')['tagID'].apply(list).to_dict()

    train_user_tag = np.zeros((len(df_train['userID']), len(df_tags['tagID'])))
    train_job_tag = np.zeros((len(df_train['jobID']), len(df_tags['tagID'])))

    # tag index padding user
    for emb_row, train_user_row in zip(train_user_tag, df_train['userID']):
        li = list(set(user_interacted_tags[train_user_row]))
        for i in li:
            emb_row[i] = 1.0

    # tag index padding job
    for emb_row, train_job_row in zip(train_job_tag, df_train['jobID']):
        li = list(set(job_interacted_tags[train_job_row]))
        for i in li:
            emb_row[i] = 1.0

    train_user_tag = torch.tensor(train_user_tag, dtype=torch.float)
    train_job_tag = torch.tensor(train_job_tag, dtype=torch.float)

    dense_ut = nn.Linear(in_features=887, out_features=128)
    user_tag_emb = dense_ut(train_user_tag)

    user_tag_emb_np = user_tag_emb.detach().numpy()
    np.save('save_emb/user_tag_emb_np', user_tag_emb_np)

    dense_jt = nn.Linear(in_features=887, out_features=128)
    job_tag_emb = dense_jt(train_job_tag)

    job_tag_emb_np = job_tag_emb.detach().numpy()
    np.save('save_emb/job_tag_emb_np', job_tag_emb_np)

if __name__ == '__main__':
    step = 2
    if step == 0:
        pre_train_emb_tag()
    elif step == 1:
        run_pretrain()
    else:
        pred = run_finetune()
        np.save('pred_ep10_neumf_emb_pretrain', np.array(pred))
        print(pred)
