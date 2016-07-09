import sframe as gl
import libffm

class FFM(object):
    def __init__(self,
                 target='target',
                 features=None,
                 num_features=2**18,
                 num_factors=4,
                 eta=0.2,
                 lambda_=0.00002,
                 num_iterations=15,
                 num_epochs=1,
                 batch_size=1,
                 num_threads=0,
                 early_stop=0,
                 verbose=True,
                 randomize=False):
        """
        Create a field-aware factorization machine model.

        Parameters
        ----------
        target : str
          The name of the column to predict. This column should be float typed.

        features : list
          The name of the feature columns that you want to use.

        num_features : int
          Number of features, including one-hot encoded features.

        num_factors : int
          Number of latent factors.

        eta : float
          Learning rate.

        lambda_ : float
          Regularization parameter.

        num_iterations : int
          The number of iterations to train the model.

        num_epochs : int
          When `batch_size > 1`, defines the number of training passes over the data.

        batch_size : int
          (not implemented)

        num_threads : int
          The number of the threads to use.

        verbose : boolean
          If true, algorithm will report progress.

        normalization : boolean
          If true, the algorithm will perform instance-wise normalization.

        random : boolean
          (not implemented) If true, the rows will be shuffled prior to training.

        References
        ----------

        - `libffm: open source C++ library
        <http://www.csie.ntu.edu.tw/~r01922136/libffm/`_
        - `FFM formulation details <http://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf>`
        - `Criteo winning submission details <http://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf>`
        """
        self.target = target
        self.features = features
        self.num_features = num_features
        self.num_factors = num_factors
        self.eta = eta
        self.lambda_ = lambda_
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.early_stop = early_stop
        self.verbose = verbose
        self.randomize = randomize

        self.m = libffm.ffm_py()
        self.m.set_params({
            'target': target,
            'features': features,
            'num_features': num_features,
            'num_factors': num_factors,
            'eta': eta,
            'lambda_': lambda_,
            'num_iterations': num_iterations,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'num_threads': num_threads,
            'early_stop': early_stop,
            'verbose': verbose,
            'randomize': randomize,
        })


    def fit(self, train, validation_set=None):
        """
        Train the model.

        Parameters
        ----------
        train : SFrame
          A training dataset containing a prediction target and feature columns
          that are dict type. Each column will be considered a "field" in
          the model. Each column element be a dictionary with integer keys
          and float values.

        validation_set : SFrame, optional
          A validation set to use for progress reporting. This should have the
          same format as the training data.

        Returns
        -------
          None

        Note
        ----
        The original library has two additional options that have not (yet)
        been exposed in this library:

        - random: rows can are processed in random order. When using this
                  wrapper, it's best to shuffle ahead of time.
        - normalization: sometimes this algorithm benefits from normalizing the
                         values row-wise. This wrapper currently requires you
                         to do that ahead of time.

        """

        train, validation_set = self._check_sframe(train, validation_set)
        self.m.fit(train, validation_set)

    def fit_partial(self, train, validation_set=None):
        train, validation_set = self._check_sframe(train, validation_set)
        self.m.fit_partial(train, validation_set)

    def _check_sframe(self, train, validation_set=None):
        if self.features is None:
            self.features = train.column_names()
            if self.target in self.features: self.features.remove(self.target)
        features = set(self.features)
        if self.target not in train.column_names():
            raise ValueError("Target column `{0}` not found in dataset.".format(self.target))
        if features & set(train.column_names()) != features:
            raise ValueError("Training data missing feature columns.")
        if validation_set is not None:
            if features & set(validation_set.column_names()) != features:
                raise ValueError("Validation data missing feature columns.")
        else:
            validation_set = train.head(0)
        return train, validation_set

    def predict(self, test):
        """
        Make predictions on a test set.

        Parameters
        ----------

        test : SFrame
          This should be in the same format as the training data. This ignores
          any columns having the same name as the target used during training.

        Returns
        -------

        out : SArray
          An SArray of predictions. This should have the same length as the
          number of rows in the provided `test` SFrame.
        """

        return self.m.predict(test)
