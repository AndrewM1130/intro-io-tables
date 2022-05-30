# The `econnet` package

## `data.py`

The dataset classes. Implement a cleaned version of the basic IO tables: the **supply** table and the **use** table.
The `Derived` class provides the intermediate quantities that allow us to construct **industry-industry** and **commodity-commodity** tables,
which are adjancency matrices for an economy graph.

The same quantities also lead to the **requirements** tables; however, those are read from a pre-calculated source for numerical reasons.

## `simulator.py`

Implements the (downstream) shock propagation as per Acemoglu et al.
The `simulateAll` method is mostly useful for sanity checks since it perpetuates the shocks created.
The `simulateOneTime` method calculates the shocks on the economy in terms of their *order* for a given initial perturbation.


