import gc

import numpy as np
import tensorflow as tf
from aeon.classification.deep_learning import (
    IndividualLITEClassifier,
    LITETimeClassifier,
)
from aeon.classification.ordinal_classification.deep_learning._ordinal_activation_layers import (
    CLM,
)
from sklearn.utils import check_random_state


class OrdinalLITETimeClassifier(LITETimeClassifier):

    def __init__(
        self,
        n_classifiers=5,
        n_filters=32,
        kernel_size=40,
        strides=1,
        activation="relu",
        file_path="./",
        save_last_model=False,
        save_best_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        batch_size=64,
        use_mini_batch_size=False,
        n_epochs=1500,
        learning_rate=0.001,
        link_function="probit",
        batch_normalization=False,
        clm_use_slope=False,
        clm_min_distance=0.35,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        optimizer=None,
    ):

        self.learning_rate = learning_rate
        self.link_function = link_function
        self.batch_normalization = batch_normalization
        self.clm_use_slope = clm_use_slope
        self.clm_min_distance = clm_min_distance

        super().__init__(
            n_classifiers=n_classifiers,
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            file_path=file_path,
            save_last_model=save_last_model,
            save_best_model=save_best_model,
            best_file_name=best_file_name,
            last_file_name=last_file_name,
            batch_size=batch_size,
            use_mini_batch_size=use_mini_batch_size,
            n_epochs=n_epochs,
            callbacks=callbacks,
            random_state=random_state,
            verbose=verbose,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
        )

    def _fit(self, X, y):
        self.classifers_ = []
        rng = check_random_state(self.random_state)

        for n in range(0, self.n_classifiers):
            cls = IndividualOrdinalLITEClassifier(
                n_filters=self.n_filters,
                kernel_size=self.kernel_size,
                file_path=self.file_path,
                save_best_model=self.save_best_model,
                save_last_model=self.save_last_model,
                best_file_name=self.best_file_name + str(n),
                last_file_name=self.last_file_name + str(n),
                batch_size=self.batch_size,
                use_mini_batch_size=self.use_mini_batch_size,
                n_epochs=self.n_epochs,
                learning_rate=self.learning_rate,
                link_function=self.link_function,
                batch_normalization=self.batch_normalization,
                clm_use_slope=self.clm_use_slope,
                clm_min_distance=self.clm_min_distance,
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


class IndividualOrdinalLITEClassifier(IndividualLITEClassifier):

    def __init__(
        self,
        n_filters=32,
        kernel_size=40,
        strides=1,
        activation="relu",
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        batch_size=64,
        use_mini_batch_size=False,
        n_epochs=1500,
        learning_rate=0.001,
        link_function="probit",
        batch_normalization=False,
        clm_use_slope=False,
        clm_min_distance=0.35,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        optimizer=None,
    ):

        self.learning_rate = learning_rate
        self.link_function = link_function
        self.batch_normalization = batch_normalization
        self.clm_use_slope = clm_use_slope
        self.clm_min_distance = clm_min_distance

        super().__init__(
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            file_path=file_path,
            save_best_model=save_best_model,
            save_last_model=save_last_model,
            best_file_name=best_file_name,
            last_file_name=last_file_name,
            batch_size=batch_size,
            use_mini_batch_size=use_mini_batch_size,
            n_epochs=n_epochs,
            callbacks=callbacks,
            random_state=random_state,
            verbose=verbose,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
        )

    def build_model(self, input_shape, n_classes, **kwargs):

        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        tf.keras.utils.set_random_seed(self.random_state_)
        input_layer, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = tf.keras.layers.Dense(1)(output_layer)
        if self.batch_normalization:
            output_layer = tf.keras.layers.BatchNormalization()(output_layer)
        output_layer = CLM(
            n_classes,
            link_function=self.link_function,
            use_slope=self.clm_use_slope,
            min_distance=self.clm_min_distance,
        )(output_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        if self.metrics is None:
            metrics = ["accuracy"]
        else:
            metrics = self.metrics

        self.optimizer_ = (
            tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            if self.optimizer is None
            else self.optimizer
        )

        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )

        return model
