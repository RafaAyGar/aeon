import gc

import numpy as np
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.ordinal_classification.deep_learning._individual_inceptionTime import (
    CustomIndividualInceptionClassifier,
)
from aeon.classification.ordinal_classification.deep_learning._ordinal_activation_layers import (
    CLM,
)
from sklearn.utils import check_random_state


class OrdinalInceptionTimeClassifier(InceptionTimeClassifier):
    def __init__(
        self,
        n_classifiers=5,
        n_filters=32,
        n_conv_per_layer=3,
        kernel_size=40,
        use_max_pooling=True,
        max_pool_size=3,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=False,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        use_custom_filters=False,
        file_path="./",
        save_last_model=False,
        save_best_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        batch_size=64,
        use_mini_batch_size=False,
        n_epochs=1500,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        optimizer=None,
        reduce_lr_on_plateau=False,
    ):
        self.reduce_lr_on_plateau = reduce_lr_on_plateau

        super().__init__(
            n_classifiers=n_classifiers,
            n_filters=n_filters,
            n_conv_per_layer=n_conv_per_layer,
            kernel_size=kernel_size,
            use_max_pooling=use_max_pooling,
            max_pool_size=max_pool_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            use_residual=use_residual,
            use_bottleneck=use_bottleneck,
            bottleneck_size=bottleneck_size,
            depth=depth,
            use_custom_filters=use_custom_filters,
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
                reduce_lr_on_plateau=self.reduce_lr_on_plateau,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
                verbose=self.verbose,
            )
            cls.fit(X, y)
            self.classifers_.append(cls)
            gc.collect()

        return self


class IndividualOrdinalInceptionClassifier(CustomIndividualInceptionClassifier):
    def __init__(
        self,
        n_filters=32,
        n_conv_per_layer=3,
        kernel_size=40,
        use_max_pooling=True,
        max_pool_size=3,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=False,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        use_custom_filters=False,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        batch_size=64,
        use_mini_batch_size=False,
        n_epochs=1500,
        callbacks=None,
        random_state=None,
        verbose=False,
        loss="categorical_crossentropy",
        metrics=None,
        optimizer=None,
        reduce_lr_on_plateau=False,
    ):
        self.reduce_lr_on_plateau = reduce_lr_on_plateau

        super().__init__(
            n_filters=n_filters,
            n_conv_per_layer=n_conv_per_layer,
            kernel_size=kernel_size,
            use_max_pooling=use_max_pooling,
            max_pool_size=max_pool_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            use_residual=use_residual,
            use_bottleneck=use_bottleneck,
            bottleneck_size=bottleneck_size,
            depth=depth,
            use_custom_filters=use_custom_filters,
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
        import numpy as np
        import tensorflow as tf

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
