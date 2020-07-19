import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from bidders import ValueDistribution, Bidder
from seller import Seller
from simulation import Simulation


np.random.seed(100)

simulations = 100
epochs = 100


df = pd.DataFrame()

for i in range(simulations):

    seller = Seller(id_=f'seller')
    value_dist = ValueDistribution(stats.beta, **{'a': 1, 'b': 2})
    bidder = Bidder(value_dist, ctr=0.3, id_=f'bidder')

    simulation = Simulation(epochs, bidder, seller)
    simulation.run()

    df_i = pd.DataFrame(
        np.array(
            [
                np.array([i for i in range(epochs)])
            ] + [
                np.array(
                    [
                        bidder.cumul_utility(t) for t in range(epochs)
                    ]
                ) for bidder in [bidder]
            ] + [
                np.array(
                    [
                        seller.cumul_revenue(t) for t in range(epochs)
                    ]
                ) for seller in [seller]
            ]
        ).T,
        columns=['time'] + [bidder.id, seller.id]
    )

    df_i['run_id'] = i
    df = df.append(df_i, ignore_index=True)
    if i % 10 == 0:
        print(i)

plt.style.use('ggplot')

fig, ax = plt.subplots(1, figsize=(10, 20))
for runid, dfi in df.groupby('run_id'):
    # bidder
    ax.plot(list(range(epochs)), dfi.iloc[:, 1].values,
            alpha=0.9,
            color='lightblue')
    # seller
    ax.plot(list(range(epochs)), dfi.iloc[:, 2].values,
            alpha=0.4,
            color='red')

dfg = df.groupby('time').agg({bidder.id: 'mean'}).reset_index()
ax.plot(
    list(range(epochs)),
    dfg[bidder.id],
    color='black',
    label=bidder.name
)
ax.legend()

dfg = df.groupby('time').agg({seller.id: 'mean'}).reset_index()
ax.plot(
    list(range(epochs)),
    dfg[seller.id],
    color='black',
    linestyle='dashed',
    label=seller.name
)
ax.legend()

plt.show()
