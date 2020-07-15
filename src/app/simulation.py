import numpy as np
import random
from app.seller import Seller
from app.bidders import Bidder


class Simulation:

    def __init__(self,
                 epochs: int,
                 bidder: Bidder,
                 seller: Seller,
                 seed: int=None):
        self.bidder = bidder
        self.seller = seller
        self.epochs = epochs
        if seed is not None:
            random.seed(seed)
        self.result = {
            'bidder': {},
            'seller': {}
        }

    def run(self):
        seller = self.seller

        for epoch in range(self.epochs):
            bidder = self.bidder
            seller.update([bidder])
            bidder.update(seller)

            assert np.isclose(
                sum(state['revenue'] for state in seller.history),
                sum(state['cost'] for state in self.bidder.history)
            )

    def get_results(self):
        bidder = self.bidder
        self.result['bidder'] = bidder.history
        self.result['seller'] = self.seller.history
        return self.result
