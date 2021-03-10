import sys
import os
import math
import logging
from itertools import combinations, product, chain
from collections import OrderedDict, defaultdict

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from functools import reduce
from bokeh.io import show, save, output_file
from bokeh.models import (
    Plot, Circle, HoverTool, TapTool,
    BoxSelectTool, ZoomInTool, ZoomOutTool,
    MultiLine, Range1d, BasicTicker, LabelSet,
    BoxZoomTool, ResetTool, ColorBar, LinearColorMapper,
    WheelZoomTool, WheelPanTool, Column,
    )
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges
from bokeh.models.transforms import CustomJSTransform
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import RangeSlider
from bokeh.palettes import Spectral4, Inferno256
from bokeh.transform import linear_cmap, transform
from bokeh.plotting import figure


logger = logging.getLogger()


def flatten(iterable):
    return list(chain.from_iterable(iterable))


NEW_BRANCH_NODE_LIMIT = 20


class SAMCTreeSearch:
    """
    Apply Monte-Carlo tree search on item combinations
    aided with the technique of simulated annealing to approximate
    the global minimum corresponding to the best fitness score.
    Here reverse fitness score is used ("price").
    Tree: hierarchical structure of branches.
    Branch: set of nodes with same length.
    Node: combination of items.
    Item: collection of main identifiers.
    Identifiers: here two hardcoded identifiers are used: (scheme, jump number).
    """

    def __init__(self, **kwargs):
        """
        :allowed_dict: dict of of possible item combinations
        :max_rounds: maximum number of rounds allowed for running
        :init_t_coeff: a coefficient for initial temperature calculation for the cooling schedule
        """
        self.allowed_dict = kwargs.get("allowed_dict")
        self.max_rounds = kwargs.get("max_rounds")
        self.init_t_coeff = kwargs.get("init_t_coeff")
        self.init_t = 0
        self.active_branches = []
        self.max_level = 0
        self.rounds = 0
        self.core_nodes = None
        G = nx.DiGraph()
        self.graph = G
        self._set_new_branch(self.allowed_dict.keys())
        self.inactive_branch_levels = set()
        self.query_pk = kwargs.get("query_pk")

    def _set_new_branch(self, nodes):
        self.max_level += 1
        try:
            parent = self.active_branches[-1]
            if parent.level == 1:
                init_t = parent.get_largest_value_diff()
                self.init_t = init_t * self.init_t_coeff
        except IndexError:
            parent = None
        branch = Branch(nodes, parent, self.core_nodes,
            self.max_level, self.graph, self.allowed_dict)
        if not branch.nodes:
            self.inactive_branch_levels.add(self.max_level)
            self.max_level -= 1
            return
        self.core_nodes = branch.core_nodes
        self.active_branches.append(branch)

    def __iter__(self):
        while self.rounds < self.max_rounds:
            self.rounds += 1
            node = self._select()
            if not node:
                #self.visualize()
                logger.info("Iteration completed in SA-MCTS")
                self.visualize_final()
                raise StopIteration
            branch = self.active_branches[-1]
            if not branch.active_nodes_for_creating:
                yield node
                logger.info("Not worth investigating further in SA-MCTS")
                self.visualize_final()
                raise StopIteration
            yield node
            if branch == node.branch and \
            (self.max_level > 1 or branch.inactive) and \
            (self.max_level + 1 not in self.inactive_branch_levels):
                self.create_new_branch()
        else:
            logger.info("Max number of rounds reached in SA-MCTS")
            self.visualize_final()
            raise StopIteration

    def _select(self):
        self._update_active_branches()
        chosen_node = None
        selection = []
        for branch in self.active_branches:
            logger.debug("Selecting new node %s at branch %s",
                self.rounds, branch)
            self._set_ucb_values(branch)
            sorted_nodes = sorted(branch.active_nodes, reverse=True,
                key=lambda n: (n.ucb_value, str(n)))
            if chosen_node:
                # solves branch jumps
                pool = chosen_node.get_relevant_descendants(
                    max_level=branch.level-1)
                chosen_node = [n for n in sorted_nodes if n in pool][0]
            else:
                chosen_node = sorted_nodes[0]
            logger.debug("Node selected: %s", chosen_node)
            selection.append(chosen_node)
            chosen_node.set_node_color("blue")

            if chosen_node.visited:
                logger.debug("Node already visited, continue")
                chosen_node.set_visits()
                continue

            chosen_node.set_visits()
            #self.visualize()

            # set color to green for further plots
            for node in selection:
                node.set_node_color("green")

            chosen_node._set_calc_round(self.rounds)
            return chosen_node

    def create_new_branch(self):
        last_active = self.active_branches[-1]
        create_from =  last_active.active_nodes_for_creating
        if create_from:
            logger.debug("Creating new branch...")
            self._set_new_branch(create_from)

    def _update_active_branches(self):
        active_branches = \
            [b for b in self.active_branches
            if not b.inactive]
        inactive_branches = \
            [b for b in self.active_branches
            if b.inactive]
        self.active_branches = active_branches
        self.inactive_branch_levels.update(
            ib.level for ib in inactive_branches)
        # keep max level updated
        if active_branches:
            self.max_level = active_branches[-1].level

    def _get_temperature(self):
        return self.init_t * float(self.max_rounds - self.rounds) / float(self.max_rounds)

    def _set_ucb_values(self, branch):
        temperature = self._get_temperature()
        branch._set_ucb_values(temperature)

    def get_info_str(self):
        return "{} rounds; T0 = {}".format(
            self.rounds, round(self.init_t, 4))

    def visualize_final(self):
        # TODO

        # consider refactor and moving of js code
        if not self.query_pk:
            return

        # Get nodes and properties
        nodes = list(self.graph)
        if not nodes:
            return
        sorted_nodes = sorted(
            [n for n in nodes if n.calc_round
            and n.price < float("inf")],
            key=lambda n: n.calc_round)
        remove_nodes = [n for n in nodes if not n.calc_round
            or n.price == float("inf")]
        new_edges = [(n1, n2)
            for n1, n2 in zip(sorted_nodes, sorted_nodes[1:])
            if len(n1) != len(n2)]
        self.graph.remove_nodes_from(remove_nodes)
        graph = nx.create_empty_copy(self.graph)
        graph.add_edges_from(new_edges, line_width=3.0)
        nodes = list(graph)
        mapping = dict([(idx, node) for idx, node in enumerate(nodes)])
        graph = nx.convert_node_labels_to_integers(graph)
        round_, name, price, halting_point, parents, branch = zip(*[
            (n.calc_round, n.full_name,
             n.price, n.halting_point,
             n.get_parent_calc_rounds(),
             n.branch.level,
            ) for n in nodes])
        max_price = math.ceil(max(price)) + 1.0 # force unequal slider ends
        min_price = math.floor(min(price))

        # Generate positions
        branch_level_mapper = defaultdict(list)
        for idx, level in enumerate(branch):
            branch_level_mapper[level].append(idx)
        branch_level_mapper = OrderedDict(branch_level_mapper)
        pos = np.zeros((len(graph), 2), dtype=float)
        y_pos = np.linspace(-1, 1, len(branch_level_mapper))
        node_count_min = 0
        node_count_max = 0
        nodelist = []
        for idx, (level, nodes) in enumerate(reversed(branch_level_mapper.items())):
            nodes = sorted(nodes, key=lambda n: mapping[n].calc_round)
            nodelist.extend(nodes)
            new_node_count = len(nodes)
            node_count_max += new_node_count
            xs = np.linspace(-1, 1, new_node_count)
            pos[node_count_min:node_count_max,0] = xs
            pos[node_count_min:node_count_max,1] = [y_pos[idx]] * new_node_count
            node_count_min += new_node_count
        pos = {node: tuple(coords) for node, coords in zip(nodelist, pos)}

        graph_renderer = from_networkx(graph, pos, scale=1, center=(0,0))
        graph_renderer.node_renderer.data_source.data['round'] = round_
        graph_renderer.node_renderer.data_source.data['name'] = name
        graph_renderer.node_renderer.data_source.data['price'] = price
        graph_renderer.node_renderer.data_source.data['parents'] = parents
        graph_renderer.node_renderer.data_source.data['halting_point'] = halting_point
        graph_renderer.node_renderer.data_source.data['selected'] = [False] * len(name)
        graph_renderer.node_renderer.data_source.data['branch'] = branch

        node_data = graph_renderer.node_renderer.data_source.data
        edge_data = graph_renderer.edge_renderer.data_source.data

        # Add labels
        code = """
            var result = new Float64Array(xs.length)
            for (var i = 0; i < xs.length; i++) {
                result[i] = provider.graph_layout[xs[i]][%s]
            }
            return result
            """
        xcoord = CustomJSTransform(
            v_func=code % "0",
            args=dict(provider=graph_renderer.layout_provider),
            )
        ycoord = CustomJSTransform(
            v_func=code % "1",
            args=dict(provider=graph_renderer.layout_provider),
            )
        # client side
        xs = transform('index', xcoord)
        ys = transform('index', ycoord)
        labels = LabelSet(
            x=xs,
            y=ys,
            text='round',
            text_font_size='8px',
            background_fill_color='white',
            text_color='black',
            source=graph_renderer.node_renderer.data_source,
            x_offset=-5,
            y_offset=-5,
            )

        # Add slider
        code = """
        var start = parseInt(cb_obj.value[0])
        var end = parseInt(cb_obj.value[1])

        function filtFunc(arr_value, arr_index, array) {
                let value = data[arr_index]
                if (value >= start && value <= end) {
                    return true
                    }
            }

        function mapFunc(arr_value, arr_index, array) {
                let value = data[arr_index]
                if (value >= start && value <= end) {
                    return arr_index
                }
            }
        var new_base = node_data[base].filter(filtFunc)
        var indices = node_data[base].map(mapFunc)
        var new_other = node_data[other].filter((v,i) => indices.includes(i))
        var new_index = node_data['index'].filter((v,i) => indices.includes(i))
        var new_name = node_data['name'].filter((v,i) => indices.includes(i))
        var new_parents = node_data['parents'].filter((v,i) => indices.includes(i))
        var new_halting_point = node_data['halting_point'].filter((v,i) => indices.includes(i))
        var new_branch = node_data['branch'].filter((v,i) => indices.includes(i))

        let new_node_data = {}
        new_node_data[base] = new_base
        new_node_data[other] = new_other
        new_node_data['index'] = new_index
        new_node_data['name'] = new_name
        new_node_data['parents'] = new_parents
        new_node_data['halting_point'] = new_halting_point
        new_node_data['branch'] = new_branch
        new_node_data['selected'] = Array(new_name.length).fill(false)

        var new_start = []
        var new_end = []
        for (i = 0; i < edge_data['start'].length; i++) {
            if (new_index.includes(edge_data['start'][i]) &&
                new_index.includes(edge_data['end'][i])) {
                    new_start.push(edge_data['start'][i])
                    new_end.push(edge_data['end'][i])
            }
        }

        let new_edge_data = {}
        new_edge_data['start'] = new_start
        new_edge_data['end'] = new_end
        new_edge_data['line_width'] = Array(new_start.length).fill(3.0)

        var branch_values = new_node_data['branch'].map((v,i,a) => a.indexOf(v) === i && [v, []])
        branch_values = branch_values.filter((v,i) => v)
        var branch_mapping = Object.fromEntries(branch_values)
        for (i = 0; i < new_node_data['branch'].length; i++) {
            branch = new_node_data['branch'][i]
            index = new_node_data['index'][i]
            branch_mapping[branch].push(index)
            }

        function linspace(length, start=-1, stop=1) {
            var ret = [];
            var step = (stop - start) / (length - 1);
            for (var i = 0; i < length; i++) {
              ret.push(start + (step * i));
                }
            return ret;
            }

        for (const [ branch, indices ] of Object.entries(branch_mapping)) {
            let branch_idx = y_pos.length - branch
            let new_y_pos = y_pos[branch_idx]
            indices.sort(function(a,b){
                return new_node_data['round'][new_node_data['index'].indexOf(a)]-
                new_node_data['round'][new_node_data['index'].indexOf(b)]
                })
            let x_pos = linspace(indices.length)
            for (i = 0; i < indices.length; i++) {
                let index = indices[i]
                let new_x_pos = x_pos[i]
                graph.layout_provider.graph_layout[index][0] = new_x_pos
                graph.layout_provider.graph_layout[index][1] = new_y_pos
                }
            }
        graph.node_renderer.data_source.data = new_node_data
        graph.edge_renderer.data_source.data = new_edge_data
        """
        callback_price = CustomJS(
            args=dict(
                graph=graph_renderer, base='price', other='round',
                data=node_data['price'],
                node_data=node_data,
                edge_data=edge_data,
                y_pos=y_pos),
            code=code)
        slider_price = RangeSlider(
            title='Price',
            start=min_price, end=max_price,
            value=(min_price, max_price),
            step=(max_price-min_price) / 50.0)
        slider_price.js_on_change('value', callback_price)

        callback_round = CustomJS(
            args=dict(
                graph=graph_renderer, base='round', other='price',
                data=node_data['round'],
                node_data=node_data,
                edge_data=edge_data,
                y_pos=y_pos),
            code=code)
        slider_round = RangeSlider(
            title='Round',
            start=min(round_), end=max(round_)+1,
            value=(min(round_), max(round_)+1),
            step=1)
        slider_round.js_on_change('value', callback_round)

        # Add tools
        code_selected = '''
            const index = cb_data.source.selected.indices[0]
            if (node_data['index'].length !== graph.node_renderer.data_source.data['index'].length) {
                    return
                }
            else if (graph.node_renderer.data_source.data['selected'][index] && true) {
                graph.node_renderer.data_source.data = {...node_data}
                graph.edge_renderer.data_source.data = {...edge_data}
                }
            else {
                var parents = node_data['parents'][index]
                parents = parents.filter((v,i) => node_data['round'].includes(v))
                var parents_index = parents.map((v,i) => {
                    const idx = node_data['round'].indexOf(v)
                    return node_data['index'][idx]
                    })

                var new_start = []
                var new_end = []
                for (p = 0; p < parents_index.length; p++) {
                    new_start.push(index)
                    new_end.push(parents_index[p])
                }
                var new_data = {}
                new_data['start'] = edge_data['start'].concat(new_start)
                new_data['end'] = edge_data['end'].concat(new_end)
                new_data['line_width'] = edge_data['line_width'].concat(Array(new_start.length).fill(0.5))
                graph.node_renderer.data_source.data['selected'] = Array(graph.node_renderer.data_source.data['selected'].length).fill(false)
                graph.node_renderer.data_source.data['selected'][index] = true
                graph.edge_renderer.data_source.data = new_data
            }
        '''
        callback_selected = CustomJS(
            args=dict(
                graph=graph_renderer,
                node_data=node_data,
                edge_data=edge_data),
            code=code_selected)
        tooltips = '''
        <div style='width:200px'>
            <span style="font-size: 15px;">Round: @round</span><br>
            <span style="font-size: 15px;">Price: @price{(0)}</span><br>
            <span style="font-size: 15px;">Parents: @parents</span><br>
            <span style="font-size: 15px;">Halt: @halting_point</span><br>
            <span style="font-size: 15px;">@name{safe}</span>
        </div>
        '''

        # Plot
        plot = Plot(
            plot_width=1000, plot_height=600,
            x_range=Range1d(-1.1,1.1),
            y_range=Range1d(-1.1,1.1),
            )
        plot.title.text = self.get_info_str()
        plot.add_tools(
            HoverTool(tooltips=tooltips),
            TapTool(callback=callback_selected), BoxSelectTool(),
            ZoomInTool(), ZoomOutTool(),
            BoxZoomTool(), ResetTool(),
            WheelZoomTool(), WheelPanTool(dimension='width'),
            WheelPanTool(dimension='height'),
            )

        # Define hovering and selection policies
        graph_renderer.selection_policy = NodesAndLinkedEdges()

        # Add colorbar
        if Inferno256[-1] != '#000003':
            #print(type(Inferno256))
            list(Inferno256).reverse()
        color_mapper = LinearColorMapper(
            palette=Inferno256,
            low=min_price,
            high=max_price,
            )
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))

        # Add node and edge shapes
        graph_renderer.node_renderer.glyph = Circle(
            size=30,
            fill_color=linear_cmap(
                'price',
                Inferno256,
                min_price,
                max_price,
                low_color='Black',
                high_color='Black',
                )
            )
        graph_renderer.node_renderer.selection_glyph = Circle(
            size=30,
            fill_color=Spectral4[2],
            )
        graph_renderer.node_renderer.hover_glyph = Circle(
            size=30,
            fill_color=Spectral4[1],
            )
        graph_renderer.edge_renderer.glyph = MultiLine(
            line_alpha=0.8,
            line_width='line_width',
            line_color='Black',
            )

        # create a plot (round vs. price)
        rounds_prices = zip(node_data['round'], node_data['price'])
        rounds_prices = sorted(rounds_prices, key=lambda t: t[0])
        rounds, prices = zip(*rounds_prices)
        tooltips = '''
        <div style='width:100px'>
            <span style="font-size: 15px;">Round: @x</span><br>
            <span style="font-size: 15px;">Price: @y{(0)}</span><br>
        </div>
        '''
        fig = figure(plot_width=1000, plot_height=350, tooltips=tooltips)
        fig.circle(rounds, prices, color='navy', alpha=0.5)
        fig.line(rounds, prices, color='navy', alpha=0.5)
        fig.xaxis.axis_label = 'Round'
        fig.yaxis.axis_label = 'Price'

        # Finalize
        plot.renderers.append(graph_renderer)
        plot.add_layout(labels)
        plot.add_layout(color_bar, 'left')
        dir_ = os.path.dirname(os.path.realpath(__file__))
        dir_ += "/templates/iq_trees/"
        try:
            os.makedirs(dir_)
        except os.error:
            pass
        output_file(dir_ + "{}.html".format(self.query_pk))
        save(Column(plot, slider_price, slider_round, fig))

    def visualize(self):
        fig = plt.gcf()
        fig.set_size_inches((30, 30))
        pos = nx.drawing.nx_pydot.graphviz_layout(self.graph, prog="dot")
        graph = self.graph
        title = "Round: {} | T: {}".format(self.rounds, self._get_temperature())
        plt.title(title)
        labels = {node: node.details for node in list(graph)}
        color_map = nx.get_node_attributes(graph, "color")
        color_list = [color_map[node] for node in graph.nodes]
        nx.draw(graph, pos, node_size=10000, with_labels=True,
            labels=labels, node_color=color_list, edgecolors="black")
        title = "Round_{}".format(self.rounds)
        #fig.savefig(title, dpi=500)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()
        plt.clf()


class Branch:

    def __init__(self, nodes, parent, core_nodes, level, graph, allowed_dict):
        """
        Nodes: list of either
        list of identifier tuples
        or
        node objects
        """
        self.parent = parent or self
        self.core_nodes = core_nodes
        self.level = level
        self.graph = graph
        self._extend_nodes(nodes, allowed_dict)

    def _extend_nodes(self, nodes, allowed_dict):
        # print(type(list(nodes)[0]))
        if type(list(nodes)[0]) == tuple:
            self.nodes = [Node(frozenset((n,)), self) for n in nodes]
            self.core_nodes = self.nodes
        elif isinstance(nodes[0], Node):
            allowed_dict = allowed_dict if self.level == 2 else None
            self.nodes = []
            agg_node_items = set()
            for core_node in self.core_nodes:
                new_nodes = [node\
                    .extend_by(core_node, self, agg_node_items, allowed_dict)
                    for node in nodes]
                agg_node_items.update([n.items for n in new_nodes if n])
                self.nodes += [n for n in new_nodes if n]
        else:
            raise TypeError("Nodes argument must be a list of either tuples\
                    or Node objects")

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        return "Level: {}, Length: {}".format(self.level, len(self))

    @property
    def inactive(self):
        return all(n.inactive_medium for n in self.nodes)

    @property
    def visits(self):
        return sum(n.visits for n in self.nodes) - len(self)

    @property
    def all_node_items(self):
        return set(n.items for n in self.nodes)

    @property
    def active_nodes(self):
        return [n for n in self.nodes if not n.inactive]

    @property
    def active_nodes_for_creating(self):
        ret = sorted(
            [n for n in self.nodes if not n.inactive_soft],
            key=lambda n: n.value, reverse=True)[:NEW_BRANCH_NODE_LIMIT]
        return ret

    def _set_ucb_values(self, temperature):
        for node in self.nodes:
            node._set_ucb_value(temperature)
            #logger.debug("Calculated UCB value for node %s at T = %s",
            #    node.details, temperature)

    def get_largest_value_diff(self):
        values = [n.value for n in self.nodes]
        return max(values) - min(values)


class Node:

    @classmethod
    def create_child_node(cls, child_items, branch):
        node = cls(child_items, branch)
        node.update(rolling_method="initialization")
        return node

    def __init__(self, items, branch):
        """
        items: frozenset of identifier tuples
        """
        self.branch = branch
        self.items = items
        self.visits = 1
        self.children = set()
        self._set_parent_nodes()
        self.value = 0
        self.price = 0
        self.update()
        # initialize ucb value in case of creating a whole new branch
        # and visualizing it without calling _set_ucb_value
        self.ucb_value = 0
        self.calc_round = 0
        self.set_node_color()
        # explicitly inactive due to bad value
        self._inactive = False
        self.halting_point = False

    def _set_calc_round(self, calc_round):
        "Set at which round this node has been chosen for exact calculation"
        self.calc_round = calc_round

    def set_node_color(self, color="white"):
        self._color = color
        self.branch.graph.nodes[self]["color"] = color

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return str([(str(jump[0]), jump[1]) for jump in self.items])

    @property
    def name_by_pks(self):
        return ", ".join([" | ".join([
            str(scheme.pk),
            str(jump),
            ]) for scheme, jump in self.items])

    @property
    def full_name(self):
        return "<br>".join([" | ".join(
            [str(scheme), str(jump)])
            for scheme, jump in self.items])

    def get_parent_calc_rounds(self):
        return [p.calc_round for p in self.parents]

    @property
    def details(self):
        return "\n".join([self.name_by_pks] +
                ["P: {}".format(round(self.price, 2)),
                 "N: {}".format(self.visits - 1),
                 "B: {}".format(self.branch.visits),
                 "V: {}".format(round(self.value, 2)),
                 "U: {}".format(round(self.ucb_value, 2))
                 ])

    def set_visits(self):
        self.visits += 1

    def update(self, rolling_method=None, value=0, base_value=1):
        if not value:
            nodes = []
            if rolling_method == "get_bulk_parents":
                # update backward, based on children, only if visited
                nodes = self.children if self.visited else []
            if rolling_method == "get_bulk_children":
                # update forward, based on parents, only if not visited
                nodes = self.parents if not self.visited else []
            if rolling_method == "initialization":
                nodes = self.parents
            if nodes:
                value = sum(n.value for n in nodes) / float(len(nodes))
        else:
        # assuming the value is price
            self.price = value
            # set the whole corresponding tree part as inactive
            if any((value > p.price + 1.0) and p.visited for p in self.parents):
                self.halting_point = True
                self._set_inactive_descendants()
            value = float(base_value) / float(value)
        self.value = value or self.value

    def _set_parent_items(self, fake_level=None):
        level = fake_level or self.branch.level - 1
        parent_items = [frozenset(c) for c in combinations(self.items, level)]
        # set only if not fake level
        if not fake_level:
            self.parent_items = parent_items
        return parent_items

    def _set_parent_nodes(self):
        self._set_parent_items()
        self.parents = set()
        # Level 1
        if not flatten(self.parent_items):
            self.branch.graph.add_node(self)
            return
        for node in self.branch.parent.nodes:
            if node.items in self.parent_items:
                self.parents.add(node)
        for p in self.parents:
            p.add_child(self)
            self.branch.graph.add_edge(p, self)

    def add_child(self, child_node):
        self.children.add(child_node)

    def get_siblings_by(self, core_node):
        parents = self.parent_items
        # dummy parents: itself
        core_parents = core_node._set_parent_items(fake_level=1)
        ret = set(reduce(frozenset.union, p) for p in product(parents, core_parents))
        return ret

    def can_be_extended_by(self, new_node, allowed_dict=None):
        # from level 1 to level 2
        if allowed_dict:
            return list(new_node.items)[0] in allowed_dict[list(self.items)[0]]
        siblings = set(s for s in self.get_siblings_by(new_node)
            if len(s) == len(self) and s != self.items)
        if siblings:
            return siblings <= self.branch.all_node_items

    def get_child_items(self, new_node, allowed_dict=None):
        if self.can_be_extended_by(new_node, allowed_dict):
            return frozenset(list(self.items) + list(new_node.items))

    def extend_by(self, new_node, branch, agg_node_items, allowed_dict=None):
        #logger.debug("Extending %s with %s", self, new_node)
        child_items = self.get_child_items(new_node, allowed_dict)
        if child_items and child_items not in agg_node_items:
            return self.__class__.create_child_node(child_items, branch)
        #child_items_str = []
        #if child_items:
        #    child_items_str = [(str(n[0]), str(n[1])) for n in child_items]
        #logger.debug("Unsuccessful extension: %s", child_items_str)

    @property
    def visited(self):
        return self.visits >= 2

    # use to select node in active branch
    @property
    def inactive(self):
        return self._inactive or self.visited and \
            (not self.children or \
            all(child.inactive for child in self.children))

    # use to determine whether corresponding branch is active
    @property
    def inactive_medium(self):
        return self._inactive or self.visited

    # use to create new branch from active nodes
    @property
    def inactive_soft(self):
        return self._inactive

    def _set_ucb_value(self, temperature):
        if len(self) == 1 and self.visits == 1:
            self.ucb_value = float("inf")
        else:
            self.ucb_value = self.value + temperature * math.sqrt(
                float(self.branch.visits) / float(self.visits)
                )

    def _rolling_update(self, value, base_value, rolling_method):
        roll = getattr(self, rolling_method)
        nodes = [self]
        while True:
            self._bulk_update(nodes, rolling_method,
                value=value, base_value=base_value)
            value = 0
            nodes = roll(nodes)
            # break if the end is reached
            if not nodes:
                break
            # also break when fully visited branch is reached
            if nodes[0].branch.inactive:
                break
            # if could not be satisfied (inf), only punish current node
            if value == float("inf"):
                break

    def rolling_update(self, value, base_value):
        rolling_methods = ["get_bulk_parents", "get_bulk_children"]
        for rolling_method in rolling_methods:
            self._rolling_update(value, base_value, rolling_method)

    def _set_inactive_descendants(self):
        self._inactive = True
        self.set_node_color("red")
        for child in self.children:
            child._set_inactive_descendants()

    def get_relevant_descendants(self, max_level):
        children = self.children
        if self.branch.level == max_level:
            return children
        return flatten([
            child.get_relevant_descendants(max_level)
            for child in children])

    @staticmethod
    def get_bulk_parents(nodes):
        return flatten(n.parents for n in nodes)

    @staticmethod
    def get_bulk_children(nodes):
        return flatten(n.children for n in nodes)

    @staticmethod
    def _bulk_update(nodes, rolling_method, value=0, base_value=1):
        for n in nodes:
            n.update(rolling_method, value=value, base_value=base_value)
