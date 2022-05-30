# -*- coding: utf-8 -*-
"""
Author: Andrej Leban
"""


# NOTE: assumes Economic_Networks is the working directory

import abc
from datetime import datetime
import json
import numpy as np
import pandas as pd
import networkx as nx


class Table(abc.ABC):
    """
    Generic IO table
    """
    mapping = json.load(open("data/category_mapping.json", 'r'))

    nInd = 71
    industries = ['Farms', 'Forestry', 'Oil & Gas', 'Mining', 'Mining activities',
                  'Utilities', 'Construction', 'Wood', 'Nonmetallic minerals',
                  'Primary metals', 'Fabricated metals', 'Machinery', 'Electronics',
                  'Electrical equipments', 'Motor Vehicles', 'Transportation equipment',
                  'Furnitures', 'Misc. Manufacturing', 'Food products', 'Textile mills',
                  'Apparel', 'Paper', 'Printing', 'Petroleum & coal', 'Chemical products',
                  'Plastics & Rubber', 'Wholesale trade', 'Vehicle dealers',
                  'Food stores', 'Merchandise stores', 'Other retail',
                  'Air transportation', 'Rail transportation', 'Water transportation',
                  'Truck transportation', 'Ground transportation',
                  'Pipeline transportation', 'Other transportation',
                  'Warehousing & storage', 'Publishing industries', 'Motion picture',
                  'Broadcasting', 'Information services', 'Federal Reserve banks',
                  'Securities', 'Insurance carriers', 'Funds & trusts', 'Housing',
                  'Real estate', 'Rental services', 'Legal services', 'Computer systems',
                  'Professional services', 'Enterprises management',
                  'Administrative services', 'Waste management', 'Educational services',
                  'Ambulatory services', 'Hospitals', 'Nursing facilities',
                  'Social assistance', 'Performing arts', 'Amusements industries',
                  'Accommodation', 'Food services', 'Other services',
                  'Government defense', 'Government nondefense',
                  'Federal Government enterprises', 'General government',
                  'Non-Federal Government enterprises']
    nComm = 73
    commodities = ['Farms', 'Forestry', 'Oil & Gas', 'Mining', 'Mining activities',
                   'Utilities', 'Construction', 'Wood', 'Nonmetallic minerals',
                   'Primary metals', 'Fabricated metals', 'Machinery', 'Electronics',
                   'Electrical equipments', 'Motor Vehicles',
                   'Transportation equipment', 'Furnitures', 'Misc. Manufacturing',
                   'Food products', 'Textile mills', 'Apparel', 'Paper', 'Printing',
                   'Petroleum & coal', 'Chemical products', 'Plastics & Rubber',
                   'Wholesale trade', 'Vehicle dealers', 'Food stores',
                   'Merchandise stores', 'Other retail', 'Air transportation',
                   'Rail transportation', 'Water transportation',
                   'Truck transportation', 'Ground transportation',
                   'Pipeline transportation', 'Other transportation',
                   'Warehousing & storage', 'Publishing industries', 'Motion picture',
                   'Broadcasting', 'Information services', 'Federal Reserve banks',
                   'Securities', 'Insurance carriers', 'Funds & trusts', 'Housing',
                   'Real estate', 'Rental services', 'Legal services',
                   'Computer systems', 'Professional services',
                   'Enterprises management', 'Administrative services',
                   'Waste management', 'Educational services', 'Ambulatory services',
                   'Hospitals', 'Nursing facilities', 'Social assistance',
                   'Performing arts', 'Amusements industries', 'Accommodation',
                   'Food services', 'Other services', 'Government defense',
                   'Government nondefense', 'Federal Government enterprises',
                   'General government', 'Non-Federal Government enterprises',
                   'Noncomparable imports and rest-of-the-world adjustment',
                   'Scrap, used and secondhand goods']

    @abc.abstractmethod
    def __init__(self, fname, indName="Name"):

        temp = pd.read_excel(fname)
        self.table = temp.iloc[6:, 1:]
        self.table.dropna(axis=0, how="all", inplace=True)

        self.table.columns = temp.iloc[5, 1:]
        self.table.rename(self.mapping, axis=1, inplace=True)

        self.table.set_index(self.table.iloc[:, 0], inplace=True)
        self.table.drop(indName, axis=1, inplace=True)

        self.table.index = np.array([Table.rename(x) for x in self.table.index.values])
        self.table.columns.name = ""
        self.table[self.table == "---"] = 0
        self.table[self.table == "..."] = 0

        self.table = self.table.astype(float)
        del temp

        # defined as property:
        # self._graph

    def make_adjacency(self):
        """
        Returns an "adjacency table" representation of the object
        """

        # this changes the ordering, let's do brute force instead
        # commonCats = set(self.table.columns).intersection(set(self.table.index.values))
        commonCats = [x for x in self.table.columns if x in self.table.index.values]

        return self.table.loc[commonCats, commonCats]

    @property
    def graph(self):
        """
        The table's underlying adjacency table as a networkX graph
        """
        if not hasattr(self, "_graph"):
            self._graph = nx.from_pandas_adjacency(self.make_adjacency())

        return self._graph

    def __str__(self):
        return self.table.__str__()

    def __repr__(self):
        return self.__str__()

    @classmethod
    def rename(cls, x):
        try:
            return cls.mapping[x]
        except Exception:
            return x


class Supply(Table):

    """
    The Supply IO Table
    """

    def __init__(self, year=2020):
        super().__init__("data/supply-tables/supply{}.xls".format(year))


class Use(Table):

    """
    The Use IO table
    """

    def __init__(self, year=2020):
        super().__init__("data/use-tables/use{}.xls".format(year))


# TODO: generalize to generic input data
class Derived(Table):

    """
    Generic class for derived tables
    """

    @abc.abstractmethod
    def __init__(self, year):

        s = Supply(year=year).table
        u = Use(year=year).table

            # the restricted supply table
        self.V = s.values[:Table.nComm, :Table.nInd]
            # the restricted use table
        self.U = u.values[:Table.nComm, :Table.nInd]

            # the total output vector by industy
        self.g = u.loc['Total industry output (basic prices)', :].values[:Table.nInd]
            # the total output vector by commodity
        self.q = s.loc[:, 'Total product supply (basic prices)'].values[:Table.nComm]

            # total use of commodities -
            # "total final demand purchases for each commodity from the use table"
        self.e = u.loc[:, 'Total use of products'].values[:Table.nComm]

            # relative amount of commodity USED by industry
        self.B = self.U @ np.diag(1 / self.g)
            # relative amount of commodity PRODUCED by industry
        self.D = self.V.T @ np.diag(1 / self.q)


class I2I(Derived):

    """
    The industry - to -industry AGGREGATE table
    """

    def __init__(self, year=2020):

        super().__init__(year)
        self.table = (self.D @ self.B)
        # TODO: check de-normalization!
        i = self.g / np.sum(self.table, axis=1)
        self.table = (self.table.T * i).T

        self.table = pd.DataFrame(data=self.table, index=Table.industries,
                                  columns=Table.industries)


class C2C(Derived):

    """
    The commodity - to -commodity AGGREGATE table
    """

    def __init__(self, year=2020):

        super().__init__(year)

        self.table = (self.B @ self.D)
        # TODO: check de-normalization!
        rel_c = np.sum(self.table, axis=1)
        rel_c[rel_c == 0] = np.finfo(float).eps
        c = self.q / rel_c
        self.table = (self.table.T * c).T

        self.table = pd.DataFrame(data=self.table, index=Table.commodities,
                                  columns=Table.commodities)


class I2IReqs(Table):

    """
    The industry - to -industry requirements table
    """

    def __init__(self, year=2020):
        super().__init__("data/i2i/i2i-reqs-{}.xls".format(year), indName="Industry Description")


class C2CReqs(Table):

    """
    The commodity - to -commodity requirements table
    """

    def __init__(self, year=2020):
        super().__init__("data/c2c/c2c-reqs-{}.xls".format(year), indName="Commodity Description")


def getEquities():
    df = pd.read_csv('data/equities/Download Data - FUND_US_ARCX_CGW.csv')[["Date", "Close"]]
    df_dim = len(df["Date"])
    df = df.rename(columns={"Close": "Water transportation"})
    df["Housing"] = pd.read_csv('data/equities/Download Data - INDEX_US_XNAS_HGX.csv')["Close"].values

    # this is missing one - I just add the same value to the end of the pandas series
    farms_series = pd.read_csv('data/equities/Download Data - INDEX_XX_S&P GSCI_SPGSAG.csv')["Close"].values
    df["Farms"] = np.append(farms_series, farms_series[-1:])

    # these have an extra date for some reason - so I decide to remove the last ones
    # that are not included in the series
    df["Truck transportation"] = pd.read_csv('data/equities/Download Data - INDEX_US__DJT.csv')["Close"].values[:df_dim]
    df["Utilities"] = pd.read_csv('data/equities/Download Data - INDEX_US__DJU.csv')["Close"].values[:df_dim]
    df["Accommodation"] = pd.read_csv('data/equities/Download Data - INDEX_XX_DOW JONES GLOBAL_DJUSLG.csv')["Close"].values[:df_dim]
    df["Chemical products"] = pd.read_csv('data/equities/Download Data - INDEX_XX_DOW JONES WILSHIRE_DWCCHM.csv')["Close"].values[:df_dim]
    df["Construction"] = pd.read_csv('data/equities/Download Data - INDEX_XX_DOW JONES WILSHIRE_DWCCNS.csv')["Close"].values[:df_dim]
    df["Machinery"] = pd.read_csv('data/equities/Download Data - INDEX_XX_DOW JONES WILSHIRE_DWCIDE.csv')["Close"].values[:df_dim]
    df["Mining"] = pd.read_csv('data/equities/Download Data - INDEX_XX_DOW JONES WILSHIRE_DWCMIN.csv')["Close"].values[:df_dim]
    df["Real estate"] = pd.read_csv('data/equities/Download Data - INDEX_XX_DOW JONES WILSHIRE_DWCRHD.csv')["Close"].values[:df_dim]
    df["Educational services"] = pd.read_csv('data/equities/Download Data - INDEX_XX_S&P US_SP1500.25302010.csv')["Close"].values[:df_dim]
    df["Air transportation"] = pd.read_csv('data/equities/Download Data - INDEX_XX_S&P US_SP500.203020.csv')["Close"].values[:df_dim]
    df["Rail transportation"] = pd.read_csv('data/equities/Download Data - INDEX_XX_S&P US_SP500.203040.csv')["Close"].values[:df_dim]
    df["Hospitals"] = pd.read_csv('data/equities/Download Data - INDEX_XX_S&P US_SP500.351020.csv')["Close"].values[:df_dim]
    df["Date"] = [datetime.strptime(date, '%m/%d/%Y') for date in list(df["Date"].values)]
    df.index = df["Date"].values
    df = df.drop(columns=["Date"])

    # some columns are strings :(
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if type(df.iloc[i, j]) != np.float64:
                df.iloc[i, j] = float((df.iloc[i, j].replace(',', '')))

    return df


if __name__ == "__main__":
    s = Supply(year=2000)
    u = Use(year=2000)
    i = I2I(2000)
    c = C2C(2000)

    ri = I2IReqs(year=2020)
    rc = C2CReqs(year=2000)

    # ri.graph

    eq = getEquities()
