from functools import partial
from copy import deepcopy
from scipy import optimize
import uuid


class Seller:

    def __init__(self, id_=None, squashing_factor=1.):
        self.id = id_ or uuid.uuid4().hex
        self.name = f'{self.__class__.__name__}-{self.id}'
        self.reserve = self._get_initial_reserve()
        self.squashing_factor = squashing_factor
        self.state = self._initialize_state()
        self.history = []
        for metric in ['revenue']:
            setattr(self, f'total_{metric}', partial(self.get_total_metric, metric))
            setattr(self, f'cumul_{metric}', partial(self.get_cumul_metric, metric))
            setattr(self, metric, partial(self.get_metric, metric))

    @staticmethod
    def _initialize_state():
        return {
            'revenue': 0.0,
            'assigned_slots': {},
            'assigned_payments': {},
            'scores': {},
            'reserve': None,
            'optimal_reserves': {}
        }

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

    @staticmethod
    def _get_initial_reserve():
        return 1.

    def get_new_reserve(self):
        old_reserve = self.reserve
        no_rev = self.state['revenue'] == 0
        new_reserve = 0.9 * old_reserve if no_rev else 1. * old_reserve
        return new_reserve

    def update(self, bidders):
        """run_auction"""
        self.state = self._initialize_state()
        self.state['reserve'] = self.reserve
        self.state['optimal_reserves'] = self.get_optimal_reserve(bidders)
        self._get_scores(bidders)
        self._assign_ad_slots()
        self._get_assigned_cpc_payments(bidders)
        self._get_revenue()
        self.history.append(deepcopy(self.state))
        self.reserve = self.get_new_reserve()

    def _get_scores(self, bidders):
        score_dict = {}
        for bidder in bidders:
            score_dict[bidder.id] = self._get_auction_score(bidder)
        self.state['scores'] = score_dict

    def _get_auction_score(self, bidder):
        if bidder.bid >= self.reserve:
            return bidder.ctr ** self.squashing_factor * bidder.bid
        else:
            return 0.

    def _assign_ad_slots(self):
        score_dict = self.state['scores']
        assigned_slots = {}
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        for position, (bidder_id, score) in enumerate(sorted_scores):
            # don't assign blocked bidders to ad slots
            assigned_slots[bidder_id] = position if score > 0. else None
        self.state['assigned_slots'] = assigned_slots

    def _get_assigned_cpc_payments(self, bidders):
        """
        first price auction
        """
        assigned_slots = self.state['assigned_slots']
        cpcs = {}
        for bidder in bidders:
            bidder_id = bidder.id
            slot = assigned_slots[bidder_id]
            if slot == 0:  # winner gets all
                cpcs[bidder_id] = bidder.ctr * bidder.bid  # 1st price auction
            else:
                cpcs[bidder_id] = 0.
        self.state['assigned_payments'] = cpcs

    def _get_revenue(self):
        state = self.state
        cpcs = state['assigned_payments']
        state['revenue'] = sum(cpcs.values())
        self.state = state

    @staticmethod
    def get_optimal_reserve(bidders):
        opt_reserves = {}
        for bidder in bidders:
            bidder_val = bidder.value_distribution
            psi = bidder.virtual_value
            assert bidder.monotone_virtual_value
            # define upper boundary value for optimize.root_scalar
            max_val = bidder_val.dist.ppf(1, **bidder_val.kwargs)
            eps = 1e-23
            while max_val == float('inf'):
                max_val = bidder_val.dist.ppf(1 - eps, **bidder_val.kwargs)
                eps *= 10
            # define x0, x1 for secant root finding method
            x0 = bidder_val.dist.ppf(0.95, **bidder_val.kwargs)
            x1 = bidder_val.dist.ppf(0.05, **bidder_val.kwargs)
            optimal_reserve = optimize.root_scalar(psi, x0=x0, x1=x1, method='secant', bracket=[0, max_val])
            opt_reserves[bidder.id] = optimal_reserve.root
        return opt_reserves
