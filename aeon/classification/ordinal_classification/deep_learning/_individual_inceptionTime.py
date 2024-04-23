import gc
import time
from copy import deepcopy

from aeon.classification.deep_learning import IndividualInceptionClassifier


class CustomIndividualInceptionClassifier(IndividualInceptionClassifier):

    def _fit(self, X, y):
        """
        Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            The training input samples of,
            shape (n_cases, n_channels, n_timepoints).
            If a 2D array-like is passed, n_channels is assumed to be 1.
        y : np.ndarray
            The training data class labels of shape (n_cases,).


        Returns
        -------
        self : object
        """
        import tensorflow as tf

        y_onehot = self.convert_y_to_keras(y)
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        if self.use_mini_batch_size:
            mini_batch_size = int(min(X.shape[0] // 10, self.batch_size))
        else:
            mini_batch_size = self.batch_size
        self.model_ = self.build_model(self.input_shape, self.n_classes_)

        if self.verbose:
            self.model_.summary()

        self.file_name_ = self.best_file_name if self.save_best_model else str(time.time_ns())

        self.callbacks_ = (
            [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                ),
                # tf.keras.callbacks.ModelCheckpoint(
                #     filepath=self.file_path + self.file_name_ + ".keras",
                #     monitor="loss",
                #     save_best_only=True,
                # ),
            ]
            if self.callbacks is None
            else self.callbacks
        )

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=mini_batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks_,
        )

        # try:
        #     self.model_ = tf.keras.models.load_model(
        #         self.file_path + self.file_name_ + ".keras", compile=False
        #     )
        #     if not self.save_best_model:
        #         os.remove(self.file_path + self.file_name_ + ".keras")
        # except FileNotFoundError:
        # self.model_ = deepcopy(self.model_)

        if self.save_last_model:
            self.save_last_model_to_file(file_path=self.file_path)

        gc.collect()
        return self
