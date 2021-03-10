#minimum working example

from sa_mcts import SAMCTreeSearch

#There are some improtant inputs:
#:allowed_dict: dictionary specifying what other items are allowed to be chosen for a given one
#:max_rounds: maximum number of rounds to be performed bby the search algorithm
#:init_t: initial temperature, the starting point of the cooling schedule is derived from this parameter

#The cooling schedule is chosen to be linear. 
#Although this is also something that should be extended to more options.

  
#Suppose there are diferent type of identifiers describing your set of items, i.e:
#item = (ID1, ID2, ..., IDn).
#In theory a combination could be built up with any number of items:
#combination = (item1, item2, ..., itemN).
#The code uses two identifiers having a "pk" attribute.
#However it could easily be made more generic.

class Identifier1:

    def __init__(self, primary_key):
        self.pk = primary_key

    def __str__(self):
        return str(self.pk)

class Identifier2:

    def __init__(self, primary_key):
        self.pk = primary_key

    def __str__(self):
        return str(self.pk)

item1 = (Identifier1(1), Identifier2(1))
item2 = (Identifier1(2), Identifier2(2))
item3 = (Identifier1(3), Identifier2(3))

options = {
    'allowed_dict': {
        item1: [item2, item3],
        item2: [item1,item3],
        item3: [item1,item2],
        },
    'max_rounds': 100,
    'init_t_coeff': 0.09,
    'query_pk': 100, #needed for bokeh html template generation
    }

iter_ = SAMCTreeSearch(**options)
node_fitness_values = range(1,100)
benchmark_fitness = 1
for idx, node in enumerate(iter_):
    # call some external functions or methods to get the fitness of this node
    # update the node and its ancestors and descendants
    node.rolling_update(node_fitness_values[idx], benchmark_fitness)
