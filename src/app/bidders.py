from abc import ABCMeta, abstractmethod
from copy import deepcopy
from functools import partial
import numpy as np
import uuid
from itertools import tee


class ValueDistribution:

    def __init__(self, dist, **kwargs):
        self.dist = dist
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def draw_sample(self):
        return self.dist.rvs(**self.kwargs)


class AbstractBidder(metaclass=ABCMeta):

    def __init__(self, value_distribution: ValueDistribution, ctr: float, id_=None):
        self.id = id_ or uuid.uuid4().hex
        self.name = f'{self.__class__.__name__}-{self.id}'
        self.value_distribution = value_distribution
        self.ctr = ctr
        self.private_value = self._get_private_value()
        self.bid = self._get_initial_bid()
        self.virtual_value = self._get_virtual_value()
        self.monotone_virtual_value = self._check_monotonicity_of_virtual_value()
        self.state = {
            'value': 0.,
            'cost': 0.,
            'utility': 0.
        }
        self.history = []
        for metric in ['value', 'cost', 'utility']:
            setattr(self, f'total_{metric}', partial(self.get_total_metric, metric))
            setattr(self, f'cumul_{metric}', partial(self.get_cumul_metric, metric))
            setattr(self, metric, partial(self.get_metric, metric))

    def _get_private_value(self):
        return self.value_distribution.draw_sample()

    def _get_virtual_value(self):
        bidder_val = self.value_distribution

        def psi(v):
            return v - (1 - bidder_val.dist.cdf(v, **bidder_val.kwargs)) / bidder_val.dist.pdf(v, **bidder_val.kwargs)

        return psi

    def _check_monotonicity_of_virtual_value(self):
        bidder_val = self.value_distribution
        psi = self.virtual_value
        support_values = np.linspace(
            bidder_val.dist.ppf(0.01, **bidder_val.kwargs),
            bidder_val.dist.ppf(0.99, **bidder_val.kwargs),
            100
        )
        return all(
            [
                psi(a) <= psi(b) for a, b in pairwise(support_values)
            ]
        )

    @abstractmethod
    def _get_initial_bid(self):
        pass

    def get_metric(self, metric, t):
        return sum([self.history[t][metric]])

    def get_total_metric(self, metric):
        print(metric)
        return sum([state[metric] for state in self.history])

    def get_cumul_metric(self, metric, t):
        try:
            return sum([state[metric] for state in self.history[:t + 1]])
        except KeyError:
            return self.get_total_metric(metric)

    def update(self, seller):
        self.state = self._get_new_state(seller)
        self._update_history()
        self.bid = self._get_new_bid()

    def _get_new_state(self, seller):
        cost = seller.state['revenue']
        state = {
            'value': self.ctr * self.private_value if cost > 0. else 0.,
            'cost': cost,
        }
        state['utility'] = state['value'] - state['cost']
        return state

    def _update_history(self):
        self.history.append(deepcopy(self.state))

    @abstractmethod
    def _get_new_bid(self):
        pass


class Bidder(AbstractBidder):

    def __init__(self, *args, **kwargs):
        super(Bidder, self).__init__(*args, **kwargs)

    def _get_initial_bid(self):
        return self.private_value / 2

    def _get_new_bid(self):
        import numpy as np
        return self.private_value / 2 * np.random.rand()


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
