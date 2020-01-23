import torch


class ClusterDist(torch.distributions.Distribution):
    def __init__(self, num_data):
        self.num_data = num_data

    def next_dist(self, z):
        """
        Args:
            z: tensor [*batch_shape, num_datapoints_so_far]
        Returns: distribution with *batch_shape [batch_size] and event_shape []
        """
        raise NotImplementedError()

    def init_z(self, sample_shape=torch.Size([])):
        return torch.zeros(*sample_shape, 0).long()

    def sample(self, sample_shape=torch.Size([])):
        z = self.next_dist(self.init_z(sample_shape)).sample()[..., None]
        for i in range(num_data - 1):
            z = torch.cat([z, self.next_dist(z).sample()[..., None]], dim=-1)
        return z

    def log_prob(self, z):
        """
        Args:
            z: tensor [*batch_shape, num_data]
        Returns: tensor [*batch_shape]
        """
        p0 = self.next_dist(self.init_z(z.shape[:-1])).log_prob(z[..., 0])
        pi = [self.next_dist(z[..., :i]).log_prob(z[..., i])
              for i in range(1, self.num_data)]
        return torch.stack([p0, *pi]).sum(dim=0)

    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def event_shape(self):
        return torch.Size([self.num_data])


class CRP(ClusterDist):
    def __init__(self, num_data):
        super().__init__(num_data)

    def next_dist(self, z):
        count = (z[:, None, :] == torch.arange(z.shape[-1] + 1)[None, :, None]).sum(dim=-1).float()
        n = (count > 0).sum(dim=-1, dtype=torch.long)
        probs_unnormalized = count + torch.nn.functional.one_hot(n, z.shape[-1] + 1)
        return torch.distributions.Categorical(probs=probs_unnormalized)


class MaskedSoftmaxClustering(ClusterDist):
    def __init__(self, logits):
        super().__init__(logits.shape[-1])
        self.logits = logits

    def next_dist(self, z):
        count = (z[:, None, :] == torch.arange(self.logits.shape[-1])[None, :, None]).sum(dim=-1).float()
        n = (count > 0).sum(dim=-1, dtype=torch.long)
        hide = (count + torch.nn.functional.one_hot(n, self.logits.shape[-1])) == 0
        logits = self.logits[..., z.shape[-1]] * 1
        logits[hide] = float("-inf")
        return torch.distributions.Categorical(logits=logits)


if __name__ == "__main__":
    batch_size = 7
    num_data = 10

    print("--- CRP ---")
    crp = CRP(num_data)
    z = crp.sample([batch_size])
    print("z =", z)
    log_prob = crp.log_prob(z)
    print("log_prob =", log_prob)

    print("\n--- Softmax ---")
    softmax_clustering = MaskedSoftmaxClustering(logits=torch.randn(batch_size, num_data, num_data))
    z = crp.sample([batch_size])
    print("z =", z)
    log_prob = crp.log_prob(z)
    print("log_prob =", log_prob)
