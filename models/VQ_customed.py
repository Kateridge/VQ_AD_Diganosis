import torch
from torch import nn
import torch.nn.functional as F


class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors, dim, beta, use_norm=False):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = dim
        self.beta = beta
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x
        # initialize codebook
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_()

    def forward(self, z, return_prob=False):
        bz, n, d = z.shape
        # l2 norm to input and codebook
        z_flattened_norm = self.norm(z.contiguous().view(-1, self.latent_dim))
        embedding_norm = self.norm(self.embedding.weight)

        d = torch.cdist(z_flattened_norm, embedding_norm, p=2)

        if return_prob:
            # calculate codebook usage prob
            vector_prob = F.softmin(d, dim=1) # (bz * a * b * c, codebook_num)
            f = lambda x: torch.sum(x, dim=0) / n
            sub_prob = torch.stack(list(map(f, torch.chunk(vector_prob, bz))), dim=0) # (bz, codebook_num)

        # quantization
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # l2 norm the input and quantized input
        z_norm = self.norm(z)
        z_q_norm = self.norm(z_q)

        # quantization loss
        loss = self.beta * torch.mean((z_q_norm.detach() - z_norm)**2) + torch.mean((z_q_norm - z_norm.detach())**2)

        z_q_norm = z + (z_q_norm - z).detach()

        if return_prob:
            codebook_usage_status = (min_encoding_indices, sub_prob)
        else:
            codebook_usage_status = min_encoding_indices
        return z_q_norm, codebook_usage_status, loss

    def get_vec_from_indice(self, indices):
        z_q = self.embedding(indices.view(-1))
        z_q = z_q.view(indices.shape[0], indices.shape[1], -1)
        return z_q

    def get_vec_from_logits(self, logits, temp=1.):
        # logits: (b, n, num_codebook)
        prob_one_hot = F.gumbel_softmax(logits, tau=temp, hard=False, dim=1) # (b, n, num_coudebook) in one-hot format
        # prob = F.softmax(logits, dim=-1).view(-1, self.num_codebook_vectors) # (b*n, num_codebook)
        embed = self.embedding.weight # (num_codebook, dim)
        vec = prob_one_hot @ embed # (b, n, dim)
        return vec



