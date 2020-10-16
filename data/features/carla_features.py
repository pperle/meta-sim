"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

from typing import Tuple

import numpy as np
from networkx import Graph

from data.features.features import Features


class CarlaFeatures(Features):
    def __init__(self, config):
        super(CarlaFeatures, self).__init__(config)

    def _encode(self, graph: Graph) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a single graph into its features (x=feature_name, y=nodes)

        @param graph: network graph
        @return: features and mask
        """
        features = []
        mask = []
        for node_idx in graph.nodes:
            node = graph.nodes[node_idx]
            node_features = []
            node_mask = []

            if 'class' in self.config['attributes']['features']:
                tmp = [0] * len(self.classes)
                if 'class' in self.config['attributes']['mutable']:
                    raise NotImplementedError

                tmp_mask = [0] * len(self.classes)
                # in this implementation classes are immutable
                idx = self.classes.index(node['cls'])
                tmp[idx] = 1

                node_mask.extend(tmp_mask)
                node_features.extend(tmp)

            for feature_name in self.config['attributes']['features']:
                if feature_name == 'class':
                    continue
                else:
                    if node['attr']['immutable']:
                        # if node is immutable
                        tmp_mask = 0
                    elif feature_name in node['attr']['mutable']:
                        # if attribute is mutable
                        tmp_mask = 1
                    else:
                        tmp_mask = 0

                    if feature_name not in node['attr']:
                        tmp = 0  # undefined value
                    else:
                        if node['attr'][feature_name + '_max'] == 0.:  # divide by 0 workaround
                            tmp = 0.
                        else:
                            tmp = node['attr'][feature_name] / node['attr'][feature_name + '_max']

                    node_mask.append(tmp_mask)
                    node_features.append(tmp)

            mask.append(node_mask)
            features.append(node_features)

        return np.array(features, dtype=np.float32), np.array(mask, dtype=np.float32)

    def _update(self, graph: Graph, feature: np.ndarray, mask: np.ndarray) -> Graph:
        """
        Update a single graph with new features

        @param graph: original graph
        @param feature_name: new features
        @param mask: mask of immutable features
        @return: graph based on `graph` and `feature`
        """
        for node_idx in graph.nodes:
            node = graph.nodes[node_idx]

            idx = 0
            if 'class' in self.config['attributes']['features']:
                idx += len(self.classes)

            for feature_name in self.config['attributes']['features']:
                if feature_name == 'class':
                    continue
                else:
                    if mask[node_idx, idx]:  # if mask is not 0 (= False)
                        if feature_name in node['attr']:  # only when `feature` is an attribute of this node
                            node['attr'][feature_name] = feature[node_idx, idx] * node['attr'][feature_name + '_max']
                            print('CarlaFeatures', feature_name, f"{feature[node_idx, idx]} * {node['attr'][feature_name + '_max']} = {node['attr'][feature_name]}")

                    idx += 1

        return graph
