from collections import defaultdict, Counter
from tqdm import tqdm
import networkx as nx

from config import UMLS_INSTALLATION_DIR, UMLS_CONCEPTS_GRAPH_FILENAME
from find_uniq_concepts import find_uniq_concepts


def yield_relations(filename):
    with open(str(filename), 'r') as f:
        for line in f:
            data = line.split('|')

            cui1 = data[0]
            cui2 = data[4]
            rel = data[3]
            rela = data[7]
            sab = data[10]

            yield (cui1, cui2, rel, rela, sab)


def calc_relations_stat(graph, selected_relations):
    relations_stat = Counter()
    relations_stat_rela = Counter()
    for u, v, d in graph.edges_iter(data=True):
        rel = d['rel']
        rela = d['rela']

        if rel in selected_relations:
            relations_stat[rel] += 1

            if rela != '':
                relations_stat_rela[rel] += 1

    return relations_stat, relations_stat_rela


def main():
    needed_cuis = find_uniq_concepts()
    print('Uniq concepts:', len(needed_cuis))

    relations_filename = UMLS_INSTALLATION_DIR / 'META/MRREL.RRF'
    relations_data = yield_relations(relations_filename)

    matched = 0
    total = 0
    graph = nx.MultiDiGraph()
    total_relations = 69188942 # this number is the number of lines in the MRREL file (`wc -l MRREL.RRF`)
    for rel_data in tqdm(relations_data, total=total_relations):
        if rel_data[0] in needed_cuis and rel_data[1] in needed_cuis:
            graph.add_edge(rel_data[0], rel_data[1], rel=rel_data[2], rela=rel_data[3], sab=rel_data[4])

            matched += 1

        total += 1

    print('Matched: {0}/{1} [{2:.2f}%]'.format(matched, total, 100 * matched / total))
    print('Graph constructed:', graph.number_of_nodes(), 'nodes', 'and', graph.number_of_edges(), 'edges')

    nx.write_gpickle(graph, str(UMLS_CONCEPTS_GRAPH_FILENAME))
    print('Graph saved:', UMLS_CONCEPTS_GRAPH_FILENAME.name)

    # selected relations
    selected_relations = {
        'CHD': 'child',
        'PAR': 'parent',
        'RB': 'broader',
        'RL': 'similar or alike',
        'RN': 'narrower',
        'RO': 'other than synonymous, narrower, or broader',
        'RQ': 'related and possibly synonymous',
        'RU': 'Related, unspecified',
    }

    relations_stat, relations_stat_rela = calc_relations_stat(graph, selected_relations)
    for rel_key, rel_name in selected_relations.items():
        print('{rel_key}\t{rel_name}\t{rel_nb}\t{rel_nb_rela}'.format(
            rel_key=rel_key, rel_name=rel_name,
            rel_nb=relations_stat[rel_key], rel_nb_rela=relations_stat_rela[rel_key]))


if __name__ == '__main__':
    main()
