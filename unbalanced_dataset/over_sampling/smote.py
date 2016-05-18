"""Class to perform over-sampling using SMOTE."""
from __future__ import print_function
from __future__ import division

import multiprocessing

import numpy as np
from numpy import concatenate

from random import betavariate

from ..unbalanced_dataset import UnbalancedDataset


class SMOTE(UnbalancedDataset):
    """Class to perform over-sampling using SMOTE.

    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique, and the variations Borderline SMOTE 1, 2 and
    SVM-SMOTE.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----
    See the original papers: [1]_, [2]_, [3]_ for more details.

    It does not support multiple classes automatically, but can be called
    multiple times.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    .. [2] H. Han, W. Wen-Yuan, M. Bing-Huan, "Borderline-SMOTE: a new
       over-sampling method in imbalanced data sets learning," Advances in
       intelligent computing, 878-887, 2005.

    .. [3] H. M. Nguyen, E. W. Cooper, K. Kamei, "Borderline over-sampling for
       imbalanced data classification," International Journal of Knowledge
       Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2001.

    """

    def __init__(self,
                 k=5,
                 m=10,
                 out_step=0.5,
                 ratio='auto',
                 random_state=None,
                 kind='regular',
                 nn_method='exact',
                 verbose=False,
                 **kwargs):
        """
        SMOTE over sampling algorithm and variations. Choose one of the
        following options: 'regular', 'borderline1', 'borderline2', 'svm'

        :param k: Number of nearest neighbours to used to construct synthetic
                  samples.

        :param m: The number of nearest neighbours to use to determine if a
                  minority sample is in danger.

        :param out_step: Step size when extrapolating

        :param ratio:
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. If an integer is given, the number of samples
            generated is equal to the number of samples in the minority class
            mulitply by this ratio.

        :param random_state: Seed for random number generation

        :param kind: The type of smote algorithm to use one of the following
                     options: 'regular', 'borderline1', 'borderline2', 'svm'

        :param nn_method: The nearest neighbors method to use which can be
                          either: 'approximate' or 'exact'. 'approximate'
                          will use LSH Forest while 'exact' will be an
                          exact search.

        :param verbose: Whether or not to print status information

        :param kwargs: Additional arguments passed to sklearn SVC object
        """

        # Parent class methods
        UnbalancedDataset.__init__(self,
                                   ratio=ratio,
                                   random_state=random_state)

        # Do not expect any support regarding the selection with this method
        if (kwargs.pop('indices_support', False)):
            raise ValueError('No indices support with this method.')

        # Get the number of processor that the user wants to use
        self.n_jobs = kwargs.pop('n_jobs', multiprocessing.cpu_count())


        # --- The type of smote
        # This object can perform regular smote over-sampling, borderline 1,
        # borderline 2 and svm smote. Since the algorithms are fairly simple
        # they share most methods.#
        self.kind = kind

        # --- Verbose
        # Control whether or not status and progress information should be#
        self.verbose = verbose

        # --- Nearest Neighbours for synthetic samples
        # The smote algorithm uses the k-th nearest neighbours of a minority
        # sample to generate new synthetic samples.#
        self.k = k

        # --- NN object
        # Import the NN object from scikit-learn library. Since in the smote
        # variations we must first find samples that are in danger, we
        # initialize the NN object differently depending on the method chosen#
        if nn_method == 'exact':
            from sklearn.neighbors import NearestNeighbors
        elif nn_method == 'approximate':
            from sklearn.neighbors import LSHForest

        if kind == 'regular':
            # Regular smote does not look for samples in danger, instead it
            # creates synthetic samples directly from the k-th nearest
            # neighbours with not filtering#
            if nn_method == 'exact':
                self.nearest_neighbour_ = NearestNeighbors(n_neighbors=k + 1,
                                                           n_jobs=self.n_jobs)
            elif nn_method == 'approximate':
                self.nearest_neighbour_ = LSHForest(n_estimators=50,
                                                    n_candidates=500,
                                                    n_neighbors=k+1)
        else:
            # Borderline1, 2 and SVM variations of smote must first look for
            # samples that could be considered noise and samples that live
            # near the boundary between the classes. Therefore, before
            # creating synthetic samples from the k-th nns, it first look
            # for m nearest neighbors to decide whether or not a sample is
            # noise or near the boundary.#
            if nn_method == 'exact':
                self.nearest_neighbour_ = NearestNeighbors(n_neighbors=m + 1,
                                                           n_jobs=self.n_jobs)
            elif nn_method == 'approximate':
                self.nearest_neighbour_ = LSHForest(n_estimators=50,
                                                    n_candidates=500,
                                                    n_neighbors=m+1)


            # --- Nearest Neighbours for noise and boundary (in danger)
            # Before creating synthetic samples we must first decide if
            # a given entry is noise or in danger. We use m nns in this step#
            self.m = m

        # --- SVM smote
        # Unlike the borderline variations, the SVM variation uses the support
        # vectors to decide which samples are in danger (near the boundary).
        # Additionally it also introduces extrapolation for samples that are
        # considered safe (far from boundary) and interpolation for samples
        # in danger (near the boundary). The level of extrapolation is
        # controled by the out_step.#
        if kind == 'svm':
            # As usual, use scikit-learn object#
            from sklearn.svm import SVC

            # Store extrapolation size#
            self.out_step = out_step

            # Store SVM object with any parameters#
            self.svm_ = SVC(**kwargs)

    def in_danger_noise(self, samples, kind='danger'):
        """Estimate if a set of sample are in danger or not.

        Parameters
        ----------
        samples : ndarray, shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        kind : str, optional (default='danger')
            The type of classification to use. Can be either:

            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        Returns
        -------
        output : ndarray, shape (n_samples, )
            A boolean array where True refer to samples in danger or noise.

        """

        # Find the NN for each samples
        # Exclude the sample itself
        x = self.nearest_neighbour_.kneighbors(samples,
                                               return_distance=False)[:, 1:]

        # Count how many NN belong to the minority class
        # Find the class corresponding to the label in x
        nn_label = (self.y[x] != self.minc).astype(int)
        # Compute the number of majority samples in the NN
        n_maj = np.sum(nn_label, axis=1)

        if kind == 'danger':
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(n_maj >= float(self.m) / 2.,
                                    n_maj < self.m)
        elif kind == 'noise':
            # Samples are noise for m = m'
            return n_maj == self.m

    def resample(self):
        """
        Main method of all children classes.

        :return: Over-sampled data set.
        """

        # Compute the ratio if it is auto
        if self.ratio == 'auto':
            self.ratio = (float(self.ucd[self.maxc] - self.ucd[self.minc]) /
                          float(self.ucd[self.minc]))


        # Start by separating minority class features and target values.
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # If regular SMOTE is to be performed#
        if self.kind == 'regular':
            # Print if verbose is true#
            if self.verbose:
                print("Finding the %i nearest neighbours..." % self.k, end="")

            # Look for k-th nearest neighbours, excluding, of course, the
            # point itself.#
            self.nearest_neighbour_.fit(minx)

            # Matrix with k-th nearest neighbours indexes for each minority
            # element.#
            nns = self.nearest_neighbour_.kneighbors(minx,
                                                     return_distance=False)[:, 1:]

            # Print status if verbose is true#
            if self.verbose:
                ##
                print("done!")

                # Creating synthetic samples #
                print("Creating synthetic samples...", end="")

            # --- Generating synthetic samples
            # Use static method make_samples to generate minority samples
            # FIX THIS SHIT!!!#
            sx, sy = self.make_samples(x=minx,
                                       nn_data=minx,
                                       y_type=self.minc,
                                       nn_num=nns,
                                       n_samples=int(self.ratio * len(miny)),
                                       step_size=1.0,
                                       random_state=self.rs,
                                       verbose=self.verbose)

            if self.verbose:
                print("done!")

            # Concatenate the newly generated samples to the original data set
            ret_x = concatenate((self.x, sx), axis=0)
            ret_y = concatenate((self.y, sy), axis=0)

            return ret_x, ret_y

        if (self.kind == 'borderline1') or (self.kind == 'borderline2'):

            if self.verbose:
                print("Finding the %i nearest neighbours..." % self.m, end="")

            # Find the NNs for all samples in the data set.
            self.nearest_neighbour_.fit(self.x)

            if self.verbose:
                print("done!")

            # Boolean array with True for minority samples in danger
            danger_index = self.in_danger_noise(minx, kind='danger')

            # If all minority samples are safe, return the original data set.
            if not any(danger_index):
                ##
                if self.verbose:
                    print('There are no samples in danger. No borderline '
                          'synthetic samples created.')

                # All are safe, nothing to be done here.#
                return self.x, self.y

            # If we got here is because some samples are in danger, we need to
            # find the NNs among the minority class to create the new synthetic
            # samples.
            #
            # We start by changing the number of NNs to consider from m + 1
            # to k + 1
            self.nearest_neighbour_.set_params(**{'n_neighbors': self.k + 1})
            self.nearest_neighbour_.fit(minx)

            # nns...#
            nns = self.nearest_neighbour_.kneighbors(minx[danger_index],
                                                     return_distance=False)[:, 1:]

            # B1 and B2 types diverge here!!!
            if self.kind == 'borderline1':
                # Create synthetic samples for borderline points.
                sx, sy = self.make_samples(minx[danger_index],
                                           minx,
                                           miny[0],
                                           nns,
                                           int(self.ratio * len(miny)),
                                           random_state=self.rs,
                                           verbose=self.verbose)

                # Concatenate the newly generated samples to the original data set
                ret_x = concatenate((self.x, sx), axis=0)
                ret_y = concatenate((self.y, sy), axis=0)

                return ret_x, ret_y

            else:
                # Split the number of synthetic samples between only minority
                # (type 1), or minority and majority (with reduced step size)
                # (type 2).
                np.random.seed(self.rs)

                # The fraction is sampled from a beta distribution centered
                # around 0.5 with variance ~0.01#
                fractions = betavariate(alpha=10, beta=10)

                # Only minority
                sx1, sy1 = self.make_samples(minx[danger_index],
                                             minx,
                                             self.minc,
                                             nns,
                                             fractions * (int(self.ratio * len(miny)) + 1),
                                             step_size=1,
                                             random_state=self.rs,
                                             verbose=self.verbose)

                # Only majority with smaller step size
                sx2, sy2 = self.make_samples(minx[danger_index],
                                             self.x[self.y != self.minc],
                                             self.minc, nns,
                                             (1 - fractions) * int(self.ratio * len(miny)),
                                             step_size=0.5,
                                             random_state=self.rs,
                                             verbose=self.verbose)

                # Concatenate the newly generated samples to the original data set
                ret_x = np.concatenate((self.x, sx1, sx2), axis=0)
                ret_y = np.concatenate((self.y, sy1, sy2), axis=0)

                return ret_x, ret_y

        if self.kind == 'svm':
            # The SVM smote model fits a support vector machine
            # classifier to the data and uses the support vector to
            # provide a notion of boundary. Unlike regular smote, where
            # such notion relies on proportion of nearest neighbours
            # belonging to each class.#

            # Fit SVM to the full data#
            self.svm_.fit(self.x, self.y)

            # Find the support vectors and their corresponding indexes
            support_index = self.svm_.support_[self.y[self.svm_.support_] == self.minc]
            support_vector = self.x[support_index]

            # First, find the nn of all the samples to identify samples in danger
            # and noisy ones
            if self.verbose:
                print("Finding the %i nearest neighbours..." % self.m, end="")

            # As usual, fit a nearest neighbour model to the data
            self.nearest_neighbour_.fit(self.x)

            if self.verbose:
                print("done!")

            # Now, get rid of noisy support vectors

            noise_bool = self.in_danger_noise(support_vector,
                                                     kind='noise')

            # Remove noisy support vectors
            support_vector = support_vector[np.logical_not(noise_bool)]

            danger_bool = self.in_danger_noise(support_vector,
                                                     kind='danger')

            # Something ...#
            safety_bool = np.logical_not(danger_bool)

            if self.verbose:
                print("Out of {0} support vectors, {1} are noisy, "
                      "{2} are in danger "
                      "and {3} are safe.".format(support_vector.shape[0],
                                                 noise_bool.sum().astype(int),
                                                 danger_bool.sum().astype(int),
                                                 safety_bool.sum().astype(int)
                                                 )
                      )

                # Proceed to find support vectors NNs among the minority class
                print("Finding the %i nearest neighbours..." % self.k, end="")

            self.nearest_neighbour_.set_params(**{'n_neighbors': self.k + 1})
            self.nearest_neighbour_.fit(minx)

            if self.verbose:
                print("done!")
                print("Creating synthetic samples...", end="")

            # Split the number of synthetic samples between interpolation and
            # extrapolation

            # The fraction are sampled from a beta distribution with mean
            # 0.5 and variance 0.01#
            np.random.seed(self.rs)
            fractions = betavariate(alpha=10, beta=10)

            # Interpolate samples in danger
            if (np.count_nonzero(danger_bool) > 0):
                nns = self.nearest_neighbour_.kneighbors(support_vector[danger_bool],
                                                         return_distance=False)[:, 1:]

                sx1, sy1 = self.make_samples(support_vector[danger_bool],
                                             minx,
                                             self.minc, nns,
                                             fractions * (int(self.ratio * len(minx)) + 1),
                                             step_size=1,
                                             random_state=self.rs,
                                             verbose=self.verbose)

            # Extrapolate safe samples
            if (np.count_nonzero(safety_bool) > 0):
                nns = self.nearest_neighbour_.kneighbors(support_vector[safety_bool],
                                                         return_distance=False)[:, 1:]
                
                sx2, sy2 = self.make_samples(support_vector[safety_bool],
                                             minx,
                                             self.minc, nns,
                                             (1 - fractions) * int(self.ratio * len(minx)),
                                             step_size=-self.out_step,
                                             random_state=self.rs,
                                             verbose=self.verbose)

            if self.verbose:
                print("done!")

            # Concatenate the newly generated samples to the original data set
            if (  (np.count_nonzero(danger_bool) > 0) and
                  (np.count_nonzero(safety_bool) > 0)     ):
                ret_x = concatenate((self.x, sx1, sx2), axis=0)
                ret_y = concatenate((self.y, sy1, sy2), axis=0)
            # not any support vectors in danger
            elif np.count_nonzero(danger_bool) == 0:
                ret_x = concatenate((self.x, sx2), axis=0)
                ret_y = concatenate((self.y, sy2), axis=0)
            # All the support vector in danger
            elif np.count_nonzero(safety_bool) == 0:
                ret_x = concatenate((self.x, sx1), axis=0)
                ret_y = concatenate((self.y, sy1), axis=0)

            return ret_x, ret_y