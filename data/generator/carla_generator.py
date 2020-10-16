"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import json

import carla

from data.features.carla_features import CarlaFeatures
from data.generator.generator import Generator
from data.renderer.carla_renderer import CarlaRenderer
from utils.carla_world import CarlaWorld


class CarlaGenerator(Generator):
    """
    Domain specific generator definition inherits the generic Generator and overrides core functions as required
    """

    def __init__(self, config: dict, host: str = '127.0.0.1', port: int = 2000):
        self.host = host
        self.port = port
        # self.rng = default_rng()
        super(CarlaGenerator, self).__init__(config, renderer=True, features=True)

    def _init_features(self) -> None:
        """
        Initializes self.features which is used when calling encode()
        """
        self.features = CarlaFeatures(self.config)

    def _init_renderer(self) -> None:
        """
        Initializes self.renderer which is used when calling render()
        """
        client = carla.Client(host=self.host, port=self.port)
        self.renderer = CarlaRenderer(self.config, CarlaWorld(client))

    def _sample_attributes(self, attr: dict, node_class: str) -> dict:
        """
        custom handling of attributes

        @param attr: dictionary of attributes from json
        @param node_class: name of the node class
        @return: dictionary of sampled attributes
        """
        # out = super(CarlaGenerator, self)._sample_attributes(attr, node_class)
        # First run to get standard attributes

        out = {}
        types = ['Deterministic', 'Gaussian']

        for key, attribute in attr.items():
            if isinstance(attribute, list) and len(attribute) > 0 and attribute[0] in types:
                value_type, values = attribute[0], attribute[1:]
                if value_type == 'Deterministic':
                    val = values[0]
                    max_value = val
                elif value_type == 'Gaussian':
                    import numpy as np
                    val = np.random.normal(values[0], values[1])
                    max_value = values[2]
                else:
                    raise ValueError('type has to be "Deterministic" or "Gaussian"')  # NotImplementedError

                # special cases
                if key == 'yaw':
                    val = val % 360  # yaw has to be between 0 and 360
                elif key == 'loc_z':
                    val = max(0, val)  # loc_z has to be higher than 0

                out[key] = val
                out[key + '_max'] = max_value
            else:
                out[key] = attribute

        return out


if __name__ == '__main__':
    config = json.load(open('config/carla.json', 'r'))
    carla_generator = CarlaGenerator(config, host='127.0.0.1', port=2000)
    sample = carla_generator.sample()
    print('sample', sample)

    import matplotlib.pyplot as plt
    from networkx import draw_networkx

    fig, axes = plt.subplots()
    draw_networkx(sample, labels={key: f"{key}-{value['cls']}" for key, value in sample._node.items()})
    plt.show()

    features_masks_list = carla_generator.encode(sample)
    adjacency_matrix = carla_generator.adjacency_matrix(sample)
    # features_masks_list[0][0][2, -1] = 0.8
    features, masks = features_masks_list[0]
    sample = carla_generator.update(sample, features, masks)
    print('sample', sample[0])

    outputs = carla_generator.render(sample)

    for output in outputs:
        image, data = output
        plt.imshow(image)
        plt.show()
