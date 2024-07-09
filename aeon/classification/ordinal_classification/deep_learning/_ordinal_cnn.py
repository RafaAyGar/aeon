import numpy as np
import tensorflow as tf
from aeon.classification.deep_learning import CNNClassifier
from aeon.classification.ordinal_classification.deep_learning._ordinal_activation_layers import (
    CLM,
)
from sklearn.utils import check_random_state


class OrdinalCNNClassifier(CNNClassifier):

    def build_model(self, input_shape, n_classes, **kwargs):

        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(1, use_bias=self.use_bias)(output_layer)
        output_layer = CLM(n_classes, link_function="logit")(output_layer)

        self.optimizer_ = (
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )
        return model
