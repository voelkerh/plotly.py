from collections import OrderedDict

from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs

# Zwischenbehelf, um die Funktionen von Bio.Phylo und io.StringIO zu importieren
from Bio import Phylo
from io import StringIO

# Optional imports, may be None for users that only use our core functionality.
np = optional_imports.get_module("numpy")
# Phylo = optional_imports.get_module("Bio.Phylo")
# StringIO = optional_imports.get_module("io.StringIO")


def create_phylogenetic_tree(
    newick_str,
    orientation="right",
):
    """
    Function that returns a phylogenetic tree Plotly figure object.

    :param (str) newick_str: Newick formatted string. Polytomy is permissible.
    :param (str) orientation: 'top', 'right', 'bottom', or 'left'

    TODO: Add examples.
    """

    if not Phylo or not StringIO:
        raise ImportError(
            "Bio.Phylo and io.StringIO are required for create_phylogenetic_tree"
        )

    phylogenetic_tree = _Phylogenetic_Tree(
        newick_str,
        orientation,
    )

    return graph_objs.Figure(
        data=phylogenetic_tree.data, layout=phylogenetic_tree.layout
    )


class _Phylogenetic_Tree(object):
    """Refer to FigureFactory.create_phylogenetic_tree() for docstring."""

    def __init__(
        self,
        newick_str,
        orientation="right",
        width=np.inf,
        height=np.inf,
        xaxis="xaxis",
        yaxis="yaxis",
    ):

        self.orientation = orientation
        self.labels = None
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.data = []
        self.leaves = []
        self.sign = {self.xaxis: 1, self.yaxis: 1}
        self.layout = {self.xaxis: {}, self.yaxis: {}}

        if self.orientation in ["left", "bottom"]:
            self.sign[self.xaxis] = 1
        else:
            self.sign[self.xaxis] = -1

        if self.orientation in ["right", "bottom"]:
            self.sign[self.yaxis] = 1
        else:
            self.sign[self.yaxis] = -1

        # Parse the Newick string
        newick_str = newick_str.replace(" ", "_")
        handle = StringIO(newick_str)
        tree = Phylo.read(handle, "newick")

        (dd_traces, ordered_labels, leaves) = self.get_phylo_tree_traces(tree)

        self.labels = ordered_labels
        self.leaves = leaves
        self.zero_vals = list(range(len(self.leaves)))

        self.layout = self.set_figure_layout(width, height)
        self.data = dd_traces

    def set_axis_layout(self, axis_key):
        """
        Sets and returns default axis object for dendrogram figure.

        :param (str) axis_key: E.g., 'xaxis', 'xaxis1', 'yaxis', yaxis1', etc.
        :rtype (dict): An axis_key dictionary with set parameters.

        """
        axis_defaults = {
            "type": "linear",
            "ticks": "",
            "mirror": "allticks",
            "rangemode": "tozero",
            "showticklabels": False,
            "zeroline": False,
            "showgrid": False,
            "showline": False,
        }

        if len(self.labels) != 0:
            axis_key_labels = self.xaxis
            if self.orientation in ["left", "right"]:
                axis_key_labels = self.yaxis
            if axis_key_labels not in self.layout:
                self.layout[axis_key_labels] = {}
            self.layout[axis_key_labels]["tickvals"] = [
                zv * self.sign[axis_key] for zv in self.zero_vals
            ]
            self.layout[axis_key_labels]["ticktext"] = self.labels
            self.layout[axis_key_labels]["tickmode"] = "array"
            self.layout[axis_key_labels]["side"] = "right"

        # Set range to invert the y-axis
        if axis_key_labels == self.yaxis:
            self.layout[axis_key_labels]["range"] = [len(self.labels) + 1, -1]

        self.layout[axis_key].update(axis_defaults)

        return self.layout[axis_key]

    def set_figure_layout(self, width, height):
        """
        Sets and returns default layout object for phylogenetic tree figure.

        """
        self.layout.update(
            {
                "showlegend": False,
                "autosize": True,
                "hovermode": "closest",
            }
        )

        self.set_axis_layout(self.xaxis)
        self.set_axis_layout(self.yaxis)

        return self.layout

    def get_phylo_tree_traces(self, tree):
        """
        Calculates all the elements needed for plotting a phylogenetic tree.

        :param (Bio.Phylo.BaseTree.Tree) tree: A Biopython Tree object parsed from a Newick formatted string.
        :rtype (tuple): Contains all the traces in the following order:
            (a) trace_list: List of Plotly trace objects for phylogenetic tree
            (b) xvals: All X points of the phylogenetic tree as array of arrays
                with length 4
            (c) yvals: All Y points of the phylogenetic tree as array of arrays
                with length 4
            (d) ordered_labels: leaf labels in the order they are going to
                appear on the plot
            (e) leaves: left-to-right traversal of the leaves

        """

        trace_list = []
        ordered_labels = []
        x_positions = {}
        y_positions = {}
        node_counter = 0

        def get_node_name(clade):
            nonlocal node_counter
            if clade.name:
                return clade.name
            else:
                node_counter += 1
                clade.name = f"internal_{node_counter}"
                return clade.name

        # Collect all terminal nodes (leaves) and their y-positions
        terminals = tree.get_terminals()
        for idx, terminal in enumerate(terminals):
            node_name = get_node_name(terminal)
            self.leaves.append(node_name)
            ordered_labels.append(node_name)
            y_positions[node_name] = idx
            x_positions[node_name] = sum(
                clade.branch_length if clade.branch_length else 1
                for clade in tree.get_path(terminal)
            )

        # Make sure all clades have names to be referenced
        for clade in tree.find_clades(order="level"):
            get_node_name(clade)

        # Traverse tree to collect y-positions for all internal nodes
        for clade in tree.find_clades(order="postorder"):
            node_name = get_node_name(clade)
            children = clade.clades
            if not clade.is_terminal():
                child_positions = [
                    y_positions[get_node_name(child)] for child in children
                ]
                y_positions[node_name] = sum(child_positions) / len(children)
            x_positions[node_name] = sum(
                clade.branch_length if clade.branch_length else 1
                for clade in tree.get_path(clade)
            )

        root_clade = tree.root
        x_positions[get_node_name(root_clade)] = 0

        # Traverse tree in level order and collect coordinates for traces
        for clade in tree.find_clades(order="level"):
            node_name = get_node_name(clade)
            children = clade.clades

            # Add a scatter trace for the node itself to display its name
            x_node = x_positions[node_name]
            y_node = y_positions[node_name]
            is_leaf = clade.is_terminal()
            trace_node = dict(
                type="scatter",
                x=[x_node],
                y=[y_node],
                mode="markers+text",
                text=[node_name] if is_leaf else [],
                hoverinfo="text",
                hovertext=[node_name] if not is_leaf else [],
                textposition="middle right",
                marker=dict(color="black", size=1),
            )
            trace_list.append(trace_node)

            for child in children:
                child_name = get_node_name(child)
                x0 = x_positions[node_name]
                x1 = x0 + (child.branch_length if child.branch_length else 1)
                y0 = y_positions[node_name]
                y1 = y_positions[child_name]

                trace1 = dict(
                    type="scatter",
                    x=[x0, x0] if self.orientation in ["left", "right"] else [y0, y0],
                    y=[y0, y1] if self.orientation in ["left", "right"] else [x0, x1],
                    mode="lines",
                    line=dict(color="black"),
                )

                trace2 = dict(
                    type="scatter",
                    x=[x0, x1] if self.orientation in ["left", "right"] else [y0, y1],
                    y=[y1, y1] if self.orientation in ["left", "right"] else [x1, x1],
                    mode="lines",
                    line=dict(color="black"),
                )

                trace_list.append(trace1)
                trace_list.append(trace2)

        return trace_list, ordered_labels, self.leaves
