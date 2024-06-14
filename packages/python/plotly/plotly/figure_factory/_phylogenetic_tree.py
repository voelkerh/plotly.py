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
    colorscale=None,
):
    """
    Function that returns a phylogenetic tree Plotly figure object.

    :param (str) newick_str: Newick formatted string. Polytomy is permissible.
    :param (str) orientation: 'top', 'right', 'bottom', or 'left'
    :param (list) colorscale: Optional colorscale for the phylogenetic tree.
                              Requires 8 colors to be specified, the 7th of
                              which is ignored.  With scipy>=1.5.0, the 2nd, 3rd
                              and 6th are used twice as often as the others.
                              Given a shorter list, the missing values are
                              replaced with defaults and with a longer list the
                              extra values are ignored.

    TODO: Add examples.
    """

    if not Phylo or not StringIO:
        raise ImportError(
            "Bio.Phylo and io.StringIO are required for create_phylogenetic_tree"
        )

    phylogenetic_tree = _Phylogenetic_Tree(
        newick_str,
        orientation,
        colorscale,
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
        colorscale=None,
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
        handle = StringIO(newick_str)
        tree = Phylo.read(handle, "newick")

        (dd_traces, xvals, yvals, ordered_labels, leaves) = self.get_phylo_tree_traces(
            tree, colorscale
        )

        self.labels = ordered_labels
        self.leaves = leaves
        yvals_flat = yvals.flatten()
        xvals_flat = xvals.flatten()

        self.zero_vals = []

        for i in range(len(yvals_flat)):
            if yvals_flat[i] == 0.0 and xvals_flat[i] not in self.zero_vals:
                self.zero_vals.append(xvals_flat[i])

        if len(self.zero_vals) > len(yvals) + 1:
            # If the length of zero_vals is larger than the length of yvals,
            # it means that there are wrong vals because of the identicial samples.
            # Three and more identicial samples will make the yvals of spliting
            # center into 0 and it will accidentally take it as leaves.
            l_border = int(min(self.zero_vals))
            r_border = int(max(self.zero_vals))
            correct_leaves_pos = range(
                l_border, r_border + 1, int((r_border - l_border) / len(yvals))
            )
            # Regenerating the leaves pos from the self.zero_vals with equally intervals.
            self.zero_vals = [v for v in correct_leaves_pos]

        self.zero_vals.sort()
        print(self.zero_vals)
        print(len(self.zero_vals))
        self.layout = self.set_figure_layout(width, height)
        self.data = dd_traces

    def get_color_dict(self, colorscale):
        """
        Returns colorscale used for phylogenetic tree clusters.

        :param (list) colorscale: Colors to use for the plot in rgb format.
        :rtype (dict): A dict of default colors mapped to the user colorscale.

        """

        # These are the color codes returned for dendrograms
        # We're replacing them with nicer colors
        # This list is the colors that can be used by dendrogram, which were
        # determined as the combination of the default above_threshold_color and
        # the default color palette (see scipy/cluster/hierarchy.py)
        d = {
            "r": "red",
            "g": "green",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
            "k": "black",
            # TODO: 'w' doesn't seem to be in the default color
            # palette in scipy/cluster/hierarchy.py
            "w": "white",
        }
        default_colors = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

        if colorscale is None:
            rgb_colorscale = [
                "rgb(0,116,217)",  # blue
                "rgb(35,205,205)",  # cyan
                "rgb(61,153,112)",  # green
                "rgb(40,35,35)",  # black
                "rgb(133,20,75)",  # magenta
                "rgb(255,65,54)",  # red
                "rgb(255,255,255)",  # white
                "rgb(255,220,0)",  # yellow
            ]
        else:
            rgb_colorscale = colorscale

        for i in range(len(default_colors.keys())):
            k = list(default_colors.keys())[i]  # PY3 won't index keys
            if i < len(rgb_colorscale):
                default_colors[k] = rgb_colorscale[i]

        # add support for cyclic format colors as introduced in scipy===1.5.0
        # before this, the colors were named 'r', 'b', 'y' etc., now they are
        # named 'C0', 'C1', etc. To keep the colors consistent regardless of the
        # scipy version, we try as much as possible to map the new colors to the
        # old colors
        # this mapping was found by inpecting scipy/cluster/hierarchy.py (see
        # comment above).
        new_old_color_map = [
            ("C0", "b"),
            ("C1", "g"),
            ("C2", "r"),
            ("C3", "c"),
            ("C4", "m"),
            ("C5", "y"),
            ("C6", "k"),
            ("C7", "g"),
            ("C8", "r"),
            ("C9", "c"),
        ]
        for nc, oc in new_old_color_map:
            try:
                default_colors[nc] = default_colors[oc]
            except KeyError:
                # it could happen that the old color isn't found (if a custom
                # colorscale was specified), in this case we set it to an
                # arbitrary default.
                default_colors[nc] = "rgb(0,116,217)"

        return default_colors

    def set_axis_layout(self, axis_key):
        """
        Sets and returns default axis object for dendrogram figure.

        :param (str) axis_key: E.g., 'xaxis', 'xaxis1', 'yaxis', yaxis1', etc.
        :rtype (dict): An axis_key dictionary with set parameters.

        """
        axis_defaults = {
            "type": "linear",
            "ticks": "outside",
            "mirror": "allticks",
            "rangemode": "tozero",
            "showticklabels": True,
            "zeroline": False,
            "showgrid": False,
            "showline": True,
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

        self.layout[axis_key].update(axis_defaults)

        return self.layout[axis_key]

    def set_figure_layout(self, width, height):
        """
        Sets and returns default layout object for phylogenetic tree figure.

        """
        self.layout.update(
            {
                "showlegend": False,
                "autosize": False,
                "hovermode": "closest",
                "width": width,
                "height": height,
            }
        )

        self.set_axis_layout(self.xaxis)
        self.set_axis_layout(self.yaxis)

        return self.layout

    def get_phylo_tree_traces(self, tree, colorscale):
        """
        Calculates all the elements needed for plotting a phylogenetic tree.

        :param (Bio.Phylo.BaseTree.Tree) tree: A Biopython Tree object parsed from a Newick formatted string.
        :param (list) colorscale: Color scale for phylogenetic tree clusters
        :rtype (tuple): Contains all the traces in the following order:
            (a) trace_list: List of Plotly trace objects for phylogenetic tree
            (b) icoord: All X points of the phylogenetic tree as array of arrays
                with length 4
            (c) dcoord: All Y points of the phylogenetic tree as array of arrays
                with length 4
            (d) ordered_labels: leaf labels in the order they are going to
                appear on the plot
            TODO: ?? (e) P['leaves']: left-to-right traversal of the leaves

        """

        trace_list = []
        colors = self.get_color_dict(colorscale)
        xvals = np.array([])
        yvals = np.array([])
        ordered_labels = []

        # Traverse tree in level order and collect coordinates for traces
        for clade in tree.find_clades(order="level"):
            if clade.is_terminal():
                self.leaves.append(clade.name)
                ordered_labels.append(clade.name)

        for clade in tree.find_clades(order="level"):
            for subclade in clade.clades:
                x0 = clade.branch_length if clade.branch_length else 0
                x1 = x0 + (subclade.branch_length if subclade.branch_length else 0)
                y = (
                    len(self.leaves) - self.leaves.index(clade.name)
                    if clade.name
                    else 0
                )
                y_sub = (
                    len(self.leaves) - self.leaves.index(subclade.name)
                    if subclade.name
                    else 0
                )

                trace = dict(
                    type="scatter",
                    x=[x0, x1] if self.orientation in ["top", "bottom"] else [y, y_sub],
                    y=[y, y_sub] if self.orientation in ["top", "bottom"] else [x0, x1],
                    mode="lines",
                    line=dict(color="black"),
                )
                trace_list.append(trace)
                xvals = np.append(xvals, [x0, x1])
                yvals = np.append(yvals, [y, y_sub])

        return trace_list, xvals, yvals, ordered_labels, self.leaves


"""
            # xs and ys are arrays of 4 points that make up the fork shapes
            for subclade in clade.clades:
                xs = [
                    clade.branch_length if clade.branch_length else 0,
                    subclade.branch_length if subclade.branch_length else 0,
                ]
                ys = [
                    clade.name if clade.name else "",
                    subclade.name if subclade.name else "",
                ]
                trace = dict(
                    type="scatter",
                    x=xs if self.orientation in ["top", "bottom"] else ys,
                    y=ys if self.orientation in ["top", "bottom"] else xs,
                    mode="lines",
                    line=dict(color="black"), # TODO: use color dictionary here?
                )
            trace_list.append(trace)
"""
