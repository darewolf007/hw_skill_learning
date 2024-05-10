from .gaussian import Gaussian

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


class ParamDict(AttrDict):
    def overwrite(self, new_params):
        for param in new_params:
            # print('overriding param {} to value {}'.format(param, new_params[param]))
            self.__setattr__(param, new_params[param])
        return self
    
def get_dim_inds(generalized_tensor):
    """ Returns a tuple 0..length, where length is the number of dimensions of the tensors"""
    return tuple(range(len(generalized_tensor.shape)))

class Loss():
    def __init__(self, weight=1.0, breakdown=None):
        """
        
        :param weight: the balance term on the loss
        :param breakdown: if specified, a breakdown of the loss by this dimension will be recorded
        """
        self.weight = weight
        self.breakdown = breakdown
    
    def __call__(self, *args, weights=1, reduction='mean', store_raw=False, **kwargs):
        """

        :param estimates:
        :param targets:
        :return:
        """
        error = self.compute(*args, **kwargs) * weights
        if reduction != 'mean':
            raise NotImplementedError
        loss = AttrDict(value=error.mean(), weight=self.weight)
        if self.breakdown is not None:
            reduce_dim = get_dim_inds(error)[:self.breakdown] + get_dim_inds(error)[self.breakdown+1:]
            loss.breakdown = error.detach().mean(reduce_dim) if reduce_dim else error.detach()
        if store_raw:
            loss.error_mat = error.detach()
        return loss
    
    def compute(self, estimates, targets):
        raise NotImplementedError
    
class NLL(Loss):
    # Note that cross entropy is an instance of NLL, as is L2 loss.
    def compute(self, estimates, targets):
        nll = estimates.nll(targets)
        return nll

class KLDivLoss(Loss):
    def compute(self, estimates, targets):
        if not isinstance(estimates, Gaussian): estimates = Gaussian(estimates)
        if not isinstance(targets, Gaussian): targets = Gaussian(targets)
        kl_divergence = estimates.kl_divergence(targets)
        return kl_divergence

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld