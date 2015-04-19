from unbalanced_dataset.unbalanced_dataset import *
__author__ = 'fmfnogueira'


class OverSampler(UnbalancedDataset):
    """
    Object to over-sample the minority class(es) by picking samples at random
    with replacement.

    *Supports multiple classes.
    """

    def __init__(self, ratio=1., random_state=None):
        """
        :param ratio:
            Number of samples to draw with respect to the number of samples in
            the original minority class.
                N_new =

        :param random_state:
            Seed.

        :return:
            Nothing.
        """
        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state)

    def resample(self):
        """
        Over samples the minority class by randomly picking samples with
        replacement.

        :return:
            overx, overy: The features and target values of the over-sampled
            data set.
        """

        # Start with the majority class
        overx = self.x[self.y == self.maxc]
        overy = self.y[self.y == self.maxc]

        # Loop over the other classes over picking at random
        for key in self.ucd.keys():
            if key == self.maxc:
                continue

            # If the ratio given is too large such that the minority becomes a
            # majority, clip it.
            if self.ratio * self.ucd[key] > self.ucd[self.maxc]:
                num_samples = self.ucd[self.maxc] - self.ucd[key]
            else:
                num_samples = int(self.ratio * self.ucd[key])

            # Pick some elements at random
            seed(self.rs)
            indx = randint(low=0, high=self.ucd[key], size=num_samples)

            # Concatenate to the majority class
            overx = concatenate((overx,
                                 self.x[self.y == key],
                                 self.x[self.y == key][indx]), axis=0)

            overy = concatenate((overy,
                                 self.y[self.y == key],
                                 self.y[self.y == key][indx]), axis=0)

        # Return over sampled dataset
        return overx, overy


class SMOTE(UnbalancedDataset):
    """
    An implementation of SMOTE - Synthetic Minority Over-sampling Technique.

    See the original paper: SMOTE - "SMOTE: synthetic minority over-sampling
    technique" by Chawla, N.V et al. for more details.

    * Does not support multiple classes automatically, but can be called
    multiple times
    """

    def __init__(self, k=5, m=10, ratio=1., kind='regular', random_state=None):
        """
        :param k:
            Number of nearest neighbours to use when constructing the synthetic
            samples.

        :param m:
            Something...

        :param ratio:
            Fraction of the number of minority samples to synthetically
            generate.

        :param kind:
            One of 'regular', 'borderline1', 'borderline2' or 'svm'.

        :param random_state:
            Seed.

        :return:
            The resampled data set with synthetic samples concatenated at the
            end.
        """

        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state)

        # Instance variable to store the number of neighbours to use.
        self.k = k

        ##
        self.m = m

    def resample(self):
        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Finding nns
        from sklearn.neighbors import NearestNeighbors

        print("Finding the %i nearest neighbours..." % self.k, end="")

        nearest_neighbour = NearestNeighbors(n_neighbors=self.k + 1)
        nearest_neighbour.fit(minx)
        nns = nearest_neighbour.kneighbors(minx, return_distance=False)[:, 1:]

        print("done!")

        # Creating synthetic samples
        print("Creating synthetic samples...", end="")

        sx, sy = self.make_samples(minx, minx, self.minc, nns,
                                   int(self.ratio * len(miny)),
                                   random_state=self.rs)
        print("done!")

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx), axis=0)
        ret_y = concatenate((self.y, sy), axis=0)

        return ret_x, ret_y


class bSMOTE1(UnbalancedDataset):
    """
    An implementation of bSMOTE type 1 - Borderline Synthetic Minority
    Over-sampling Technique - type 1.

    See the original paper: "Borderline-SMOTE: A New Over-Sampling Method in
    Imbalanced Data Sets Learning,
    by Hui Han, Wen-Yuan Wang, Bing-Huan Mao" for more details.

    * Does not support multiple classes automatically, but can be called
    multiple times
    """

    def __init__(self, k=5, m=10, ratio=1., random_state=None):
        """
        :param k:
            The number of nearest neighbours to use to construct the synthetic
            samples.

        :param m:
            The number of nearest neighbours to use to determine if a minority
            sample is in danger.

        :param ratio:
            Fraction of the number of minority samples to synthetically
            generate.

        :param random_state:
            Seed.

        :return:
            The resampled data set with synthetic samples concatenated at the
            end.
        """
        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state)

        # NN for synthetic samples
        self.k = k
        # NN for in_danger?
        self.m = m

    def resample(self):
        from sklearn.neighbors import NearestNeighbors

        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Find the NNs for all samples in the data set.
        print("Finding the %i nearest neighbours..." % self.m, end="")
        nn = NearestNeighbors(n_neighbors=self.m + 1)
        nn.fit(self.x)

        print("done!")

        # Boolean array with True for minority samples in danger
        index = [self.in_danger(x, self.y, self.m, miny[0], nn) for x in minx]
        index = asarray(index)

        # If all minority samples are safe, return the original data set.
        if not any(index):
            print('There are no samples in danger. No borderline synthetic '
                  'samples created.')
            return self.x, self.y

        # Find the NNs among the minority class
        nn.set_params(**{'n_neighbors': self.k + 1})
        nn.fit(minx)
        nns = nn.kneighbors(minx[index], return_distance=False)[:, 1:]

        # Create synthetic samples for borderline points.
        sx, sy = self.make_samples(minx[index], minx, miny[0], nns,
                                   int(self.ratio * len(miny)),
                                   random_state=self.rs)

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx), axis=0)
        ret_y = concatenate((self.y, sy), axis=0)

        return ret_x, ret_y


class bSMOTE2(UnbalancedDataset):
    """
    An implementation of bSMOTE type 2 - Borderline Synthetic Minority
    Over-sampling Technique - type 2.

    See the original paper: "Borderline-SMOTE: A New Over-Sampling Method in
    Imbalanced Data Sets Learning,
    by Hui Han, Wen-Yuan Wang, Bing-Huan Mao" for more details.

    * Does not support multiple classes automatically, but can be called
    multiple times
    """

    def __init__(self, k=5, m=10, ratio=1., random_state=None):
        """
        :param k:
            The number of nearest neighbours to use to construct the synthetic
            samples.

        :param m:
            The number of nearest neighbours to use to determine if a minority
            sample is in danger.

        :param ratio:
            Fraction of the number of minority samples to synthetically
            generate.

        :param random_state:
            Seed.

        :return:
            The resampled data set with synthetic samples concatenated at the
            end.
        """

        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state)

        # NN for synthetic samples
        self.k = k

        # NN for in_danger?
        self.m = m

    def resample(self):
        from sklearn.neighbors import NearestNeighbors

        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        # Find the NNs for all samples in the data set.
        print("Finding the %i nearest neighbours..." % self.m, end="")
        nn = NearestNeighbors(n_neighbors=self.m + 1)
        nn.fit(self.x)

        print("done!")

        # Boolean array with True for minority samples in danger
        index = [self.in_danger(x, self.y, self.m, self.minc, nn) for x in minx]
        index = asarray(index)

        # If all minority samples are safe, return the original data set.
        if not any(index):
            print('There are no samples in danger. No borderline synthetic '
                  'samples created.')
            return self.x, self.y

        # Find the NNs among the minority class
        nn.set_params(**{'n_neighbors': self.k + 1})
        nn.fit(minx)
        nns = nn.kneighbors(minx[index], return_distance=False)[:, 1:]

        # Split the number of synthetic samples between only minority
        # (type 1), or minority and majority (with reduced step size)
        # (type 2).
        pyseed(self.rs)
        fractions = min(max(gauss(0.5, 0.1), 0), 1)

        # Only minority
        sx1, sy1 = self.make_samples(minx[index], minx, self.minc, nns,
                                     fractions * (int(self.ratio * len(miny)) + 1),
                                     step_size=1,
                                     random_state=self.rs)

        # Only majority with smaller step size
        sx2, sy2 = self.make_samples(minx[index], self.x[self.y != self.minc],
                                     self.minc, nns,
                                     (1 - fractions) * int(self.ratio * len(miny)),
                                     step_size=0.5,
                                     random_state=self.rs)

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx1, sx2), axis=0)
        ret_y = concatenate((self.y, sy1, sy2), axis=0)

        return ret_x, ret_y


class SVM_SMOTE(UnbalancedDataset):
    """
    Implementation of support vector borderline SMOTE.

    Similar to borderline SMOTE it only created synthetic samples for
    borderline samples, however it looks for borderline samples by fitting and
    SVM classifier and identifying the support vectors.

    See the paper: "Borderline Over-sampling for Imbalanced Data
    Classification, by Nguyen, Cooper, Kamei"

    * Does not support multiple classes, however it can be called multiple
    times (I believe).
    """

    def __init__(self, k=5, m=10, out_step=0.5, ratio=1, random_state=None,
                 **kwargs):
        """

        :param k:
            Number of nearest neighbours to used to construct synthetic
            samples.

        :param m:
            The number of nearest neighbours to use to determine if a minority
            sample is in danger.

        :param ratio:
            Fraction of the number of minority samples to synthetically
            generate.

        :param out_step:
            Step size when extrapolating


        :param svm_args:
            Arguments to pass to the scikit-learn SVC object.

        :param random_state:
            Seed

        :return:
            Nothing.
        """

        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state)

        self.k = k
        self.m = m
        self.out_step = out_step

        ##
        from sklearn.svm import SVC
        self.svm = SVC(**kwargs)

    def resample(self):
        """
        ...
        """

        from sklearn.neighbors import NearestNeighbors

        # Fit SVM and find the support vectors
        self.svm.fit(self.x, self.y)
        support_index = self.svm.support_[self.y[self.svm.support_] == self.minc]
        support_vector = self.x[support_index]

        # Start with the minority class
        minx = self.x[self.y == self.minc]

        # First, find the nn of all the samples to identify samples in danger
        # and noisy ones
        print("Finding the %i nearest neighbours..." % self.m, end="")
        nn = NearestNeighbors(n_neighbors=self.m + 1)
        nn.fit(self.x)
        print("done!")

        # Now, get rid of noisy support vectors

        # Boolean array with True for noisy support vectors
        noise_bool = []
        for x in support_vector:
            noise_bool.append(self.is_noise(x, self.y, self.minc, nn))

        # Turn into array#
        noise_bool = asarray(noise_bool)

        # Remove noisy support vectors
        support_vector = support_vector[logical_not(noise_bool)]

        # Find support_vectors there are in danger (interpolation) or not
        # (extrapolation)
        danger_bool = [self.in_danger(x, self.y, self.m, self.minc, nn)
                       for x in support_vector]

        # Turn into array#
        danger_bool = asarray(danger_bool)

        #Something ...#
        safety_bool = logical_not(danger_bool)

        #things to print#
        print_stats = (len(support_vector),
                       noise_bool.sum(),
                       danger_bool.sum(),
                       safety_bool.sum())

        print("Out of %i support vectors, %i are noisy, %i are in danger "
              "and %i are safe." % print_stats)

        # Proceed to find support vectors NNs among the minority class
        print("Finding the %i nearest neighbours..." % self.k, end="")
        nn.set_params(**{'n_neighbors': self.k + 1})
        nn.fit(minx)
        print("done!")

        print("Creating synthetic samples...", end="")

        # Split the number of synthetic samples between interpolation and
        # extrapolation
        pyseed(self.rs)
        fractions = min(max(gauss(0.5, 0.1), 0), 1)

        # Interpolate samples in danger
        nns = nn.kneighbors(support_vector[danger_bool],
                            return_distance=False)[:, 1:]

        sx1, sy1 = self.make_samples(support_vector[danger_bool], minx,
                                     self.minc, nns,
                                     fractions * (int(self.ratio * len(minx)) + 1),
                                     step_size=1,
                                     random_state=self.rs)

        # Extrapolate safe samples
        nns = nn.kneighbors(support_vector[safety_bool],
                            return_distance=False)[:, 1:]

        sx2, sy2 = self.make_samples(support_vector[safety_bool], minx,
                                     self.minc, nns,
                                     (1 - fractions) * int(self.ratio * len(minx)),
                                     step_size=-self.out_step,
                                     random_state=self.rs)

        print("done!")

        # Concatenate the newly generated samples to the original data set
        ret_x = concatenate((self.x, sx1, sx2), axis=0)
        ret_y = concatenate((self.y, sy1, sy2), axis=0)

        return ret_x, ret_y