import gc

import numpy as np
import tensorflow as tf
from aeon.classification.deep_learning import (
    InceptionTimeClassifier,
    IndividualInceptionClassifier,
)
from aeon.classification.ordinal_classification.deep_learning._ordinal_activation_layers import CLM
from sklearn.utils import check_random_state


class OrdinalInceptionTimeClassifier(InceptionTimeClassifier):

    def _fit(self, X, y):
        self.classifers_ = []
        rng = check_random_state(self.random_state)

        for n in range(0, self.n_classifiers):
            cls = IndividualOrdinalInceptionClassifier(
                n_filters=self.n_filters,
                n_conv_per_layer=self.n_conv_per_layer,
                kernel_size=self.kernel_size,
                use_max_pooling=self.use_max_pooling,
                max_pool_size=self.max_pool_size,
                strides=self.strides,
                dilation_rate=self.dilation_rate,
                padding=self.padding,
                activation=self.activation,
                use_bias=self.use_bias,
                use_residual=self.use_residual,
                use_bottleneck=self.use_bottleneck,
                depth=self.depth,
                use_custom_filters=self.use_custom_filters,
                file_path=self.file_path,
                save_best_model=self.save_best_model,
                save_last_model=self.save_last_model,
                best_file_name=self.best_file_name + str(n),
                last_file_name=self.last_file_name + str(n),
                batch_size=self.batch_size,
                use_mini_batch_size=self.use_mini_batch_size,
                n_epochs=self.n_epochs,
                callbacks=self.callbacks,
                loss=self.loss,
                metrics=self.metrics,
                optimizer=self.optimizer,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
                verbose=self.verbose,
            )
            cls.fit(X, y)
            self.classifers_.append(cls)
            gc.collect()

        return self


class IndividualOrdinalInceptionClassifier(IndividualInceptionClassifier):

    def build_model(self, input_shape, n_classes, **kwargs):
        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(1)(output_layer)
        output_layer = tf.keras.layers.BatchNormalization()(output_layer)
        output_layer = CLM(n_classes, link_function="logit")(output_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics

        self.optimizer_ = (
            tf.keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

        return model
