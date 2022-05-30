# -*- coding: utf-8 -*-
"""
Author: Andrej Leban
"""

from collections import OrderedDict
import copy
import os

import numpy as np
import pandas as pd

import networkx as nx

import uuid

import matplotlib.pyplot as plt
import seaborn as sns

from econnet import data, deltaz


# naive implementation for testing purposes, keep propagating self-effects
def simulateAll(y, H, dz, niter=20, linearUnit=True):
    """
    Self-perpetuating perturbation

    Parameters
    ----------
    y : The outputs in ABSOLUTE UNITS (not relative).
    H : The Leontieff inverse.
    dz :The perturbation, either log or linear. See linear Unit.
    niter : optional
        The number of iterations. The default is 20.
    linearUnit : optional
        Whether dz is to be interpreted as a relative change. The default is True.
        If true, a 25% decrease is given as dz = -0.25.
        If false, it is log(1 - 0.25) =  -0.288...

    Returns
    -------
    ys : A list of successive ys (columns) in the simulation

    """
    if linearUnit:
        dz = np.log(1 + dz)
    ys = [y]

    for i in range(niter):
        dly = H @ dz
        y = np.exp(np.log(y) + dly)
        ys.append(y)

        # the previous effect becomes the shock
        dz = dly

    return ys


def simulateOneTime(y, H, dz, niter=100, linearUnit=True):
    """One-time shock
       The iterations correspond to the ORDER of perturbation effects

    Parameters
    ----------
    y : The outputs in ABSOLUTE UNITS (not relative)..
    H : The Leontieff inverse.
    dz :The perturbation, either log or linear. See linear Unit..
    niter : optional
        Max order calulated. Mostly irrelevant here since it exits early. The default is 100.
    linearUnit : optional
        Whether dz is to be interpreted as a relative change. The default is True.
        If true, a 25% decrease is given as dz = -0.25.
        If false, it is log(1 - 0.25) =  -0.288...

    Returns
    -------
    ys : A list of successive ys (columns) in the simulation

    """

    if linearUnit:
        dz = np.log(1 + dz)
    ys = [y]
    seen = set()

    for i in range(niter):
        dly = H @ dz
        y = np.exp(np.log(y) + dly)
        ys.append(y)

        # TODO: use an eps cutoff instead of strict 0?
        # the shock propagates downstream
        seen.update([i for i in np.where(dz)[0]])

        if len(seen) == len(y):
            break
        if sum(dly) == 0:
            break

        # only set the ones that were perturbed for the first time
        # as the new perturbation
        dz = np.zeros(dz.shape)
        mask = sorted(set(range(len(y))) - seen)
        dz[mask] = dly[mask]

    return ys


def toLinear(dx):
    return np.log(1 + dx)


def toLog(dx):
    return np.exp(dx) - 1


def simulateOneTimePlot(year, dz, niter=100, linearUnit=True):

    if linearUnit:
        dz = toLog(dz)

    seen = set()

    u = data.Use(year)
    y = u.table.loc["Value Added (basic prices)", :][:data.Table.nInd]

    ys = [y]

    IR = data.I2IReqs(year)
    H = IR.make_adjacency()
    # edges are proportional to the requirements
    G = IR.graph

    # NOTE: the attributes are log-scale
    for node in G.nodes:
        G.nodes[node]["y"] = y[node]
        G.nodes[node]["dly"] = 0.0

    for i in range(niter):

        # NOTE: creating separate plots for cause - dz- and effect - dly
        for node in G.nodes:
            G.nodes[node]["dz"] = dz[node]
            #NOTE: disabling this to keep the "echo" in the graph
            # G.nodes[node]["dly"] = 0.0

        plotGraph(G)

        dly = H @ dz
        y = np.exp(np.log(y) + dly)
        ys.append(y)

        # Effect
        for node in G.nodes:
            G.nodes[node]["y"] = y[node]
            G.nodes[node]["dly"] = dly[node]

        plotGraph(G)

        # the shock propagates downstream
        seen.update([i for i in np.where(dz)[0]])

        if len(seen) == len(y):
            break
        if sum(dly) == 0:
            break

        # only set the ones that were perturbed for the first time
        # as the new perturbation
        dz[:] = 0.0
        mask = sorted(set(range(len(y))) - seen)
        dz[mask] = dly[mask]

    return ys


def plotGraph(G):

    G = copy.deepcopy(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    dlys = pd.Series(nx.get_node_attributes(G, "dly")).sort_values(ascending=True)
    dzs = pd.Series(nx.get_node_attributes(G, "dz")).sort_values(ascending=True)

    #ONLY KEEP TOP  dlys, dzs
    newdlys = dlys[dlys != 0]
    newdlys = newdlys.iloc[:15]
    newdzs = dzs[dzs != 0]
    newdzs = newdzs.iloc[:10]

    subgraph = list(set(newdlys.index) | set(newdzs.index) |
                    plotGraph._nodes)
    if len(subgraph) == 0:
        return

    plotGraph._nodes |= set(subgraph)

    G2 = G.subgraph(subgraph)

    pos = nx.spring_layout(G2,
                           pos=plotGraph._pos,
                           fixed=plotGraph._pos)

    if plotGraph._pos is None:
        plotGraph._pos = pos
    else:
        plotGraph._pos = {**plotGraph._pos, **pos}

    # size by y
    node_size = [1e-3 * nx.get_node_attributes(G2, "y")[v] for v in G2]

    # edge width by H
    edge_width = [5e1 * G2[u][v]['weight'] for u, v in G2.edges()]

    # Color
    node_color = pd.Series(np.array(["grey"] * (len(subgraph))),
                           index=subgraph)
    node_color[newdlys.index] = "orange"
    node_color[newdzs.index] = "red"

    # Alpha

    alphas = pd.Series(0, index=subgraph)
    # dz has precedence for those with both
    alphas[newdlys.index] = newdlys.values
    alphas[newdzs.index] = newdzs.values

    # Transform for clearer visualization
    alphas = np.abs(alphas.apply(toLinear))
    alphasTr = (alphas * 5)**3
    alphasTr[alphasTr != 0] = alphasTr[alphasTr != 0] / np.max(alphasTr[alphasTr != 0])
    alphasTr = np.clip(alphasTr + 0.1, 0, 1)

    plt.figure()

    nx.draw_networkx_edges(
        G=G2,
        pos=plotGraph._pos,
        width=edge_width,
        alpha=0.15
    )

    nx.draw_networkx_nodes(
        G=G2,
        pos=plotGraph._pos,
        node_size=node_size,
        node_color=node_color,
        alpha=alphasTr.values
    )

    nx.draw_networkx_labels(
        G=G2,
        pos=plotGraph._pos,
        font_size=8,
        alpha=0.75
    )

    if plotGraph._dirname is None:
        plotGraph._dirname = "figs/" + str(uuid.uuid4())[:6]
        if not os.path.exists(plotGraph._dirname):
            os.makedirs(plotGraph._dirname)

    plt.savefig(plotGraph._dirname + "/" + str(plotGraph._i),
                dpi=300)
    plotGraph._i += 1


plotGraph._pos = None
plotGraph._nodes = set()
plotGraph._dirname = None
plotGraph._i = 0

if __name__ == "__main__":

    # # initial state - using VALUE ADDED
    # u = data.Use()
    # y = u.table.loc["Value Added (basic prices)", :][:data.Table.nInd]
    # ys = []

    # H = data.I2IReqs().make_adjacency()

    dz = deltaz.equityDz(data.getEquities())
    ys = simulateOneTimePlot(2020, dz)

    # # dz = np.zeros(y.shape[0])
    # # dz[np.random.randint(0, y.shape[0])] = - np.random.rand()
    # # let's shock housing - it doesn't make sense to shock more than exp(-0.5)
    # # = 40% decrease
    # # dz[48] = - np.random.uniform(0, 0.5)

    # all_run = simulateAll(y=y, H=H, dz=dz)
    # ar = np.array(all_run)

    # #NOTE: Housing -index 48 - is upstream from real estate and unaffected!!!
    # sns.heatmap((ar / ar[0, :]).T)
    # plt.xlabel("time")
    # plt.axhline(np.where(dz), color="red")
    # plt.title("Self-perpetuating shocks (relative)")

    # # Delta function shock
    # one_run = simulateOneTime(y=y, H=H, dz=dz)
    # oar = np.array(one_run)

    # plt.figure()
    # plt.plot(oar / oar[0, :], marker="x")
    # plt.xlabel("effect order")
    # plt.ylabel("proportion of inital value")
    # plt.yscale('log')
    # plt.title("Delta-function shock effects (relative)")
