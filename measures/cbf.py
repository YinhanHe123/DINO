import random
import networkx as nx

from measures.base_measure import BaseMeasure

class CBFinder(BaseMeasure):
    def __init__(self, *args) :
        super().__init__("CBFinder")
        
    def get_immunized_nodes(self, A, k):
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        selected_nodes = []
        while len(selected_nodes) < k:
            visited_nodes = []
            # Start at a random node
            node = random.choice(list(G.nodes))
            visited_nodes.append(node)
            
            while True: # walk through unseen neighbors of node
                neighbors = [n for n in G.neighbors(node) if n not in visited_nodes]
                if not neighbors:
                    break

                node = random.choice(neighbors)
                visited_nodes.append(node)

                # Check if there is more than one connection from 'node' to any of the visited nodes
                # Every node will have at least one connection to the previous node, so we're looking for additional connections
                back_connections = [n for n in G.neighbors(node) if n in visited_nodes]
                if len(back_connections) == 1:
                    # Potential target identified
                    potential_target = back_connections[0]

                    # Pick two random neighboring nodes of 'node' (other than the potential target)
                    other_neighbors = [n for n in G.neighbors(node) if n != potential_target]
                    if len(other_neighbors) < 2:
                        continue

                    other_neighbors = random.sample(other_neighbors, 2)

                    # Check for connections back to the previously visited nodes
                    back_connections_other = [n for n in G.neighbors(other_neighbors[0]) if n in visited_nodes] + \
                                            [n for n in G.neighbors(other_neighbors[1]) if n in visited_nodes]

                    if not back_connections_other:
                        # No such connections exist - immunize the potential target
                        if potential_target not in selected_nodes:
                            selected_nodes.append(potential_target)
                        break

                # Check if we have reached the desired number of immunized nodes
                if len(selected_nodes) == k:
                    return selected_nodes
        return selected_nodes
