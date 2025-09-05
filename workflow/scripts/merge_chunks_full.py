"""
Merge chunks into a tree sequence per chromosome and MCMC sample.

Part of https://github.com/nspope/singer-snakemake.
"""

try:
    import msprime

    NODE_IS_RE_EVENT = msprime.NODE_IS_RE_EVENT
except ImportError:
    # Hard-coded constant at v'1.3.4'
    NODE_IS_RE_EVENT = 131072
import os
import subprocess
import tskit
import pickle
import numpy as np
import yaml
import json
import tszip
from datetime import datetime


# --- lib --- #


def tag():
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


def pipeline_provenance(version_string, parameters):
    git_dir = os.path.join(snakemake.scriptdir, os.path.pardir, os.path.pardir, ".git")
    git_commit = subprocess.run(
        ["git", f"--git-dir={git_dir}", "describe", "--always"],
        capture_output=True,
    )
    if git_commit.returncode == 0:
        git_commit = git_commit.stdout.strip().decode("utf-8")
        version_string = f"{version_string}.{git_commit}"
    return {
        "software": {"name": "singer-snakemake", "version": version_string},
        "parameters": snakemake.config,
        "environment": tskit.provenance.get_environment(),
    }


def singer_provenance(version_string, parameters):
    return {
        "software": {"name": "singer", "version": version_string},
        "parameters": parameters,
        "environment": tskit.provenance.get_environment(),
    }


def add_recombination(nodes, edges, breakpoints, lower, upper, time):
    # First, we make sure time is monotonically increasing
    time = enforce_non_decreasing(time)
    # I'm not sure if SINGER produces COMMON_ANCESTRY events,
    # which should be handle differently. Here, I check if that's
    # the case and raise an error
    assert np.all(lower != upper), "Common ancestry event founded"
    # Now, we process every recombination event that affected
    # each node in increasing order
    lex_order = np.lexsort((time, lower))
    # TODO: check if it's more efficient to remove extra edges every time
    # or it's worthy to remove them all at once
    keep_edges = list(np.ones(edges.num_rows, dtype=bool))
    for i in lex_order:
        # First, we add both nodes that identify this event
        # in trees to the left and to the right of the breakpoint
        recomb1 = nodes.add_row(flags=NODE_IS_RE_EVENT, time=time[i])
        recomb2 = nodes.add_row(flags=NODE_IS_RE_EVENT, time=time[i])
        # Now, we find the edges we have to delete
        mask_child = edges.child == lower[i]
        # Because we are iterating in sorted order, the last edge is it's
        # always the last one we added
        edge_left = np.where((edges.right == breakpoints[i]) & mask_child)[0][-1]
        edge_right = np.where((edges.left == breakpoints[i]) & mask_child)[0][-1]
        # Edges to the left
        cur_lower = lower[i]
        parent = edges.parent[edge_left]
        while nodes.flags[parent] == NODE_IS_RE_EVENT:
            # Because of the same reasons as above
            cur_lower = parent
            # Find the edge
            edge_left = np.where(
                (edges.child == cur_lower) & (edges.right == breakpoints[i])
            )[0][0]
            parent = edges.parent[edge_left]
        assert parent == upper[i]
        # We remove this edge that goes from lower to upper
        # and add one edge that goes from lower to recomb1 and from
        # recomb1 to upper
        left_interval = (edges.left[edge_left], edges.right[edge_left])
        edges.add_row(
            child=cur_lower,
            parent=recomb1,
            left=left_interval[0],
            right=left_interval[1],
        )
        edges.add_row(
            child=recomb1,
            parent=upper[i],
            left=left_interval[0],
            right=left_interval[1],
        )
        assert nodes.time[cur_lower] < nodes.time[recomb1] < nodes.time[upper[i]]
        keep_edges.extend([True, True])
        # Edges to the right
        right_interval = (edges.left[edge_right], edges.right[edge_right])
        parent = edges.parent[edge_right]
        cur_lower = lower[i]
        while nodes.flags[parent] == NODE_IS_RE_EVENT:
            # Because of the same reasons as above
            cur_lower = parent
            # Find the edge
            edge_right = np.where(
                (edges.child == cur_lower) & (edges.left == breakpoints[i])
            )[0][0]
            parent = edges.parent[edge_right]
        edges.add_row(
            child=cur_lower,
            parent=recomb2,
            left=right_interval[0],
            right=right_interval[1],
        )
        edges.add_row(
            child=recomb2,
            parent=parent,
            left=right_interval[0],
            right=right_interval[1],
        )
        assert nodes.time[cur_lower] < nodes.time[recomb2] < nodes.time[parent]
        keep_edges.extend([True, True])
        keep_edges[edge_left] = False
        keep_edges[edge_right] = False
    assert len(keep_edges) == edges.num_rows, "Mismatch in edge length!"
    edges.keep_rows(keep_edges)
    # tables.sort()


def enforce_non_decreasing(time):
    # First, argsort
    sorted_idx = np.argsort(time)
    prev = -1
    for i in sorted_idx:
        time[i] = max(np.nextafter(prev, np.inf), time[i])
        prev = time[i]
    return time


# --- implm --- #

min_branch_length = 1e-7  # TODO: make settable?
stratify = snakemake.params.stratify

ratemap = pickle.load(open(snakemake.input.ratemap, "rb"))
metadata = pickle.load(open(snakemake.input.metadata, "rb"))

tables = tskit.TableCollection(sequence_length=ratemap.sequence_length)
nodes, edges, individuals, populations = (
    tables.nodes,
    tables.edges,
    tables.individuals,
    tables.populations,
)

parameters = []
population_map = {}
num_nodes, num_samples = 0, 0
files = zip(snakemake.input.params, snakemake.input.recombs)
keep_edges = []  # To discard edges later
for i, (params_file, recomb_file) in enumerate(files):
    node_file = recomb_file.replace("_recombs_", "_nodes_")
    mutation_file = recomb_file.replace("_recombs_", "_muts_")
    branch_file = recomb_file.replace("_recombs_", "_branches_")
    params = yaml.safe_load(open(params_file))
    block_start = params["start"]
    parameters.append(params)
    # nodes
    node_time = np.loadtxt(node_file)
    num_nodes = nodes.num_rows - num_samples
    if individuals.num_rows == 0:
        population = []
        num_samples = np.sum(node_time == 0.0)
        individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
        populations.metadata_schema = tskit.MetadataSchema.permissive_json()
        for meta in metadata:
            individuals.add_row(metadata=meta)
            if stratify in meta:  # recode as integer
                population_name = meta[stratify]
                if not population_name in population_map:
                    population_map[population_name] = len(population_map)
                    populations.add_row(metadata={"name": population})
                population.append(population_map[population_name])
            else:
                population.append(-1)
        ploidy = num_samples / individuals.num_rows
        assert ploidy == 1.0 or ploidy == 2.0
        for i in range(num_samples):
            individual = i // int(ploidy)
            nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE,
                population=population[individual],
                individual=individual,
            )
    min_time = 0
    for t in node_time:  # NB: nodes are sorted, ascending in time
        if t > 0.0:
            # TODO: assertion triggers rarely (FP error?)
            # assert t >= min_time
            t = max(min_time + min_branch_length, t)
            nodes.add_row(time=t)
            min_time = t

    # edges
    edge_span = np.loadtxt(branch_file)
    edge_span = edge_span[edge_span[:, 2] >= 0, :]
    length = max(edge_span[:, 1])
    parent_indices = np.array(edge_span[:, 2], dtype=np.int32)
    child_indices = np.array(edge_span[:, 3], dtype=np.int32)
    parent_indices[parent_indices >= num_samples] += num_nodes
    child_indices[child_indices >= num_samples] += num_nodes
    edges.append_columns(
        left=edge_span[:, 0] + block_start,
        right=edge_span[:, 1] + block_start,
        parent=parent_indices,
        child=child_indices,
    )
    keep_edges.extend([True] * len(parent_indices))

    # mutations
    mutations = np.loadtxt(mutation_file)
    num_mutations = mutations.shape[0]
    mut_pos = 0
    for i in range(num_mutations):
        if mutations[i, 0] != mut_pos and mutations[i, 0] < length:
            tables.sites.add_row(
                position=mutations[i, 0] + block_start,
                ancestral_state="0",
            )
            mut_pos = mutations[i, 0]
        site_id = tables.sites.num_rows - 1
        mut_node = int(mutations[i, 1])
        if mut_node < num_samples:
            tables.mutations.add_row(
                site=site_id,
                node=int(mutations[i, 1]),
                derived_state=str(int(mutations[i, 3])),
            )
        else:
            tables.mutations.add_row(
                site=site_id,
                node=int(mutations[i, 1]) + num_nodes,
                derived_state=str(int(mutations[i, 3])),
            )
    # recombinations
    # Read recombination data from file
    recombs = np.loadtxt(recomb_file)
    # We have to correct the the data
    breakpoints = recombs[:, 0].astype(int) + block_start
    lower = recombs[:, 1].astype(int)
    lower[lower >= num_samples] += num_nodes
    upper = recombs[:, 2].astype(int)
    upper[upper >= num_nodes] += num_nodes
    time = recombs[:, 3].astype(float)
    time = enforce_non_decreasing(time)
    add_recombination(nodes, edges, breakpoints, lower, upper, time)


# rebuild mutations table in time order at each position
mut_time = tables.nodes.time[tables.mutations.node]
mut_coord = tables.sites.position[tables.mutations.site]
mut_order = np.lexsort((-mut_time, mut_coord))
mut_state = tskit.unpack_strings(
    tables.mutations.derived_state,
    tables.mutations.derived_state_offset,
)
mut_state, mut_state_offset = tskit.pack_strings(np.array(mut_state)[mut_order])
tables.mutations.set_columns(
    site=tables.mutations.site[mut_order],
    node=tables.mutations.node[mut_order],
    time=np.repeat(tskit.UNKNOWN_TIME, tables.mutations.num_rows),
    derived_state=mut_state,
    derived_state_offset=mut_state_offset,
)

tables.provenances.add_row(
    json.dumps(
        pipeline_provenance(snakemake.params.version["pipeline"], snakemake.config)
    )
)
tables.provenances.add_row(
    json.dumps(singer_provenance(snakemake.params.version["singer"], parameters))
)

tables.sort()
tables.build_index()
tables.compute_mutation_parents()
ts = tables.tree_sequence()
tszip.compress(ts, snakemake.output.trees)
