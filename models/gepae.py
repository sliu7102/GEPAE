r"""
    @Author : sliu
    @README : 
    @Date   : 2024-10-08 16: 14: 30
    @Related: 
"""


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch_geometric
import scipy.sparse as sp


from models.transformer import ViT
from models.wrapper import preprocess_item
from models.ViTdecoder import vitdecoder
from models.config import Config
import models.spatial as S




def get_graph_info(file_name, channel_num, instance_len, patchsize):


    if instance_len % patchsize !=0: raise ValueError(f"wrong instance len")
    factor = instance_len // patchsize


    prior = pd.read_excel(f'./{file_name}', header=None) 
    adj_matrix = prior.fillna(0).iloc[1:, 1:].values
    rows, cols = adj_matrix.shape
    if rows != cols and rows!=channel_num: raise ValueError("\n****************************  wrong prior knowledge  ****************************\n")
    

    adj_matrix = np.repeat(adj_matrix, factor, axis=0) 
    adj_matrix = np.repeat(adj_matrix, factor, axis=1) 

    adj = sp.coo_matrix(adj_matrix)
    indices = np.vstack((adj.row, adj.col)) 
    edge_index = torch.LongTensor(indices) 
    num_edges = edge_index.shape[1]
    num_nodes = int(channel_num * factor)
    fea_length = 1024

    edge_attr = torch.randint(1, 2, (num_edges, 3), dtype=torch.int) 

    x = torch.randn(num_nodes, fea_length)

    item = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr= edge_attr)
    batched_data = preprocess_item(item)

    return batched_data




class Restormer(nn.Module):
    def __init__(
        self,
        channel_num: int,
        image_size: tuple,
        patch_size: tuple,
        dim: int,
        decoder_initial_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        file_name: str, ## fault: None
    ):
        super(Restormer, self).__init__()

        self.dim = dim
        self.decoder_initial_dim = decoder_initial_dim
        self.channel_num = channel_num
        self.patchsize = patch_size[1]
        self.instance_len = image_size[1]
        self.file_name = file_name
        self.factor = 16


        self.encoder = ViT( ## encoding
            image_size=image_size,
            patch_size=patch_size,
            dim=self.dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            node_num=int(self.channel_num * self.instance_len // self.patchsize),
        )
        assert channel_num == image_size[0], 'Channle number is not equal !'


        self.decoder = vitdecoder(      ## decoding
            out_channels=1,
            dim=self.dim,
            initial_dim= self.decoder_initial_dim,
            channels=self.channel_num,
            num_blocks=[1, 1, 2, 2],  
            num_refinement_blocks= 2, 
            heads=[1, 2, 4, 8], 
            ffn_expansion_factor= 12, 
        )

        self.Digcap = S.DigitCaps(
            in_num_caps=(image_size[0] * image_size[1] // patch_size[0] // patch_size[1])*self.factor, 
            in_dim_caps= self.dim//self.factor,
            out_dim_caps= self.dim, 
            
        )

        self.conf = Config()
        if self.conf.USE_CUDA:
            self.mask = torch.ones(1, image_size[0]//patch_size[0], image_size[1]//patch_size[1]).bool().cuda()
        else:
            self.mask = torch.ones(1, image_size[0]//patch_size[0], image_size[1]//patch_size[1]).bool()

        if self.training:
            print("\nInitializing network weights.........")
            initialize_weights(self.encoder, self.decoder)

    def forward(self, x):
        b = x.size(0)                                   ## batch size
        batched_data = get_graph_info(self.file_name, self.channel_num, self.instance_len, self.patchsize) 

        encoded = self.encoder(x, batched_data, self.mask) ## bs x (channel_num * patch num) x dim

        if self.training:
            encoded = add_noise(encoded, noise_type="gaussian", sd=0.2) 
            print("\nAdding noise.........")

        encoded1, vectors = self.Digcap(encoded.view(b, encoded.size(1)*self.factor, -1))
        recons = self.decoder(encoded1.permute(0, 2, 1).unsqueeze(2)) ## B x dim x 1 x 1  --> B x 1 x channel_num x instance_len

        return encoded, recons, vectors.squeeze(1)


## Initialize weight function
def initialize_weights(*models): 
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def add_noise(latent, noise_type="gaussian", sd=0.2): ##### Adding Noise ############
    """Here we add noise to the latent features concatenated from the 4 autoencoders.
    Arguements:
    'gaussian' (string): Gaussian-distributed additive noise.
    'speckle' (string) : Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.
    'sd' (integer) : standard deviation used for geenrating noise

    Input :
        latent : numpy array or cuda tensor.

    Output:
        Array: Noise added input, can be np array or cuda tnesor.
    """
    conf = Config()
    assert sd >= 0.0
    if noise_type == "gaussian":
        mean = 0.

        n = torch.distributions.Normal(torch.tensor([mean]), torch.tensor([sd]))
        if conf.USE_CUDA:
            noise = n.sample(latent.size()).squeeze(-1).cuda()
        else:
            noise = n.sample(latent.size()).squeeze(-1)
        latent = latent + noise
        return latent

    if noise_type == "speckle":
        if conf.USE_CUDA:
            noise = torch.randn(latent.size()).cuda()
        else:
            noise = torch.randn(latent.size())
        latent = latent + latent * noise
        return latent



if __name__ == "__main__":
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")


    channel_num = 12
    instance_len = 1024
    patch_size = 128
    dim = 384
    decoder_initial_dim = 64

    file_relation = "relationship.xlsx"


    model = Restormer(
        channel_num=channel_num,
        image_size=(channel_num, instance_len),
        patch_size=(1, patch_size), 
        dim=dim, 
        decoder_initial_dim= decoder_initial_dim,
        depth=6,
        heads=8,
        mlp_dim=1024,
        file_name=file_relation,
    ).to(device)


