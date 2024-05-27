from importlib import import_module

from measures.base_measure import BaseMeasure

MEASURE_MAP = {'acquaintance': 'Acquaintance', 'betweenness_centrality': 'BetweennessCentrality', 
               'cbf': 'CBFinder', 'closeness_centrality': "ClosenessCentrality", 'contain': "Contain", 'degree_direct': 'DegreeDirect',
               'dino': 'Dino', 'hits': 'Hits', 'katz_centrality': 'KatzCentrality', 'page_rank':'PageRank', 'random':'Random',
               'greedy':'Greedy'}

def get_largest_dict_keys(dict, k):
    sorted_score_pairs = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    largest_keys = [pair[0] for pair in sorted_score_pairs[:k]]
    return largest_keys

def get_measure(measure_type, args) -> BaseMeasure:
    module = import_module(f"measures.{measure_type}")
    measure_class = getattr(module, MEASURE_MAP[measure_type])
    return measure_class(args)