import graphlab as gl
import libffm

class FFM(object):
    def __init__(self, eta=.1, k=4, lam=0.0):
        """
        Create a field-aware factorization machine model.

        Parameters
        ----------
        eta : float
          Learning rate.

        k : int
          Number of latent factors.

        lam : float
          Regularization parameter.


        References
        ----------

        - `libffm: open source C++ library
        <http://www.csie.ntu.edu.tw/~r01922136/libffm/`_
        - `FFM formulation details <http://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf>`
        - `Criteo winning submission details <http://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf>`
        """
        self.m = libffm.ffm_py()
        self.m.init_model(eta, lam, k)

    def fit(self, train, validation_set=None,
            target='target', features=None,
            max_feature_id=2**18,
            nr_iters=15, nr_threads=1,
            quiet=False):
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

        target : str
          The name of the column to predict. This column should be float typed.

        features : list
          The name of the feature columns that you want to use.

        nr_iters : int
          The number of iterations to train the model.

        nr_threads : int
          The number of the threads to use.

        quiet : boolean
          If true, algorithm will report progress.

        normalization : boolean
          If true, the algorithm will perform instance-wise normalization.

        random : boolean
          If true, the rows will be shuffled prior to training.

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

        if target not in train.column_names():
            raise ValueError, "Target column `{0}` not found in dataset.".format(target)
        if validation_set is not None:
            if train.column_names() != validation_set.column_names():
                raise ValueError, "Train, validation data must have the same column names."
        else:
            validation_set = train.head(0)
        if features is None:
            features = [c for c in train.column_names() if c is not target]
        self.m.set_param(nr_iters, nr_threads, quiet)
        self.m.fit(train, validation_set, target, features, max_feature_id)

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
