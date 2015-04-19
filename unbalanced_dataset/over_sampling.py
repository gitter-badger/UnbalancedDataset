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
            x_oversampled, y_oversampled: The features and target values of the over-sampled
            data set.
        """

        # Start with the majority class
        x_oversampled = self.x[self.y == self.maxc]
        y_oversampled = self.y[self.y == self.maxc]

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
            numpy.random.seed(self.rs)
            indx = numpy.random.randint(low=0, high=self.ucd[key], size=num_samples)

            # Concatenate to the majority class
            x_oversampled = concatenate((x_oversampled,
                                         self.x[self.y == key],
                                         self.x[self.y == key][indx]), axis=0)

            y_oversampled = concatenate((y_oversampled,
                                         self.y[self.y == key],
                                         self.y[self.y == key][indx]), axis=0)

        # Return over sampled dataset
        return x_oversampled, y_oversampled


class SMOTE(UnbalancedDataset):
    """
    Reg
    An implementation of SMOTE - Synthetic Minority Over-sampling Technique.

    See the original paper: SMOTE - "SMOTE: synthetic minority over-sampling
    technique" by Chawla, N.V et al. for more details.

    * Does not support multiple classes automatically, but can be called
    multiple times

    B1
    An implementation of bSMOTE type 1 - Borderline Synthetic Minority
    Over-sampling Technique - type 1.

    See the original paper: "Borderline-SMOTE: A New Over-Sampling Method in
    Imbalanced Data Sets Learning,
    by Hui Han, Wen-Yuan Wang, Bing-Huan Mao" for more details.

    * Does not support multiple classes automatically, but can be called
    multiple times

    B2
    An implementation of bSMOTE type 2 - Borderline Synthetic Minority
    Over-sampling Technique - type 2.

    See the original paper: "Borderline-SMOTE: A New Over-Sampling Method in
    Imbalanced Data Sets Learning,
    by Hui Han, Wen-Yuan Wang, Bing-Huan Mao" for more details.

    * Does not support multiple classes automatically, but can be called
    multiple times

    SVM
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
                 kind='regular', verbose=0,
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

        ##
        self.kind = kind

        ##
        self.verbose = verbose

        ##
        self.k = k

        ##
        from sklearn.neighbors import NearestNeighbors

        if kind == 'regular':
            self.nearest_neighbour_ = NearestNeighbors(n_neighbors=k + 1)
        ##
        else:
            ##
            self.nearest_neighbour_ = NearestNeighbors(n_neighbors=m + 1)

            ##
            self.m = m

        ##
        if kind == 'svm':
            from sklearn.svm import SVC
            self.out_step = out_step
            self.svm_ = SVC(**kwargs)

    def resample(self):
        # Start with the minority class
        minx = self.x[self.y == self.minc]
        miny = self.y[self.y == self.minc]

        if self.kind == 'regular':
            ##
            if self.verbose:
                print("Finding the %i nearest neighbours..." % self.k, end="")

            ##
            self.nearest_neighbour_.fit(minx)
            nns = self.nearest_neighbour_.kneighbors(minx,
                                                     return_distance=False)[:, 1:]

            if self.verbose:
                ##
                print("done!")

                # Creating synthetic samples #
                print("Creating synthetic samples...", end="")

            sx, sy = self.make_samples(minx, minx, self.minc, nns,
                                       int(self.ratio * len(miny)),
                                       random_state=self.rs)

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
            danger_index = [self.in_danger(x, self.y, self.m, miny[0],
                            self.nearest_neighbour_) for x in minx]

            # Turn into numpy array#
            danger_index = asarray(danger_index)

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

            #B1 and B2 diverge here!!!#

            if self.kind == 'borderline1':
                # Create synthetic samples for borderline points.
                sx, sy = self.make_samples(minx[danger_index], minx, miny[0], nns,
                                           int(self.ratio * len(miny)),
                                           random_state=self.rs)

                # Concatenate the newly generated samples to the original data set
                ret_x = concatenate((self.x, sx), axis=0)
                ret_y = concatenate((self.y, sy), axis=0)

                return ret_x, ret_y

            else:
                # Split the number of synthetic samples between only minority
                # (type 1), or minority and majority (with reduced step size)
                # (type 2).
                numpy.random.seed(self.rs)

                # The fraction is sampled from a beta distribution centered
                # around 0.5 with variance ~0.1#
                fractions = betavariate(alpha=10, beta=10)

                # Only minority
                sx1, sy1 = self.make_samples(minx[danger_index], minx, self.minc, nns,
                                             fractions * (int(self.ratio * len(miny)) + 1),
                                             step_size=1,
                                             random_state=self.rs)

                # Only majority with smaller step size
                sx2, sy2 = self.make_samples(minx[danger_index], self.x[self.y != self.minc],
                                             self.minc, nns,
                                             (1 - fractions) * int(self.ratio * len(miny)),
                                             step_size=0.5,
                                             random_state=self.rs)

                # Concatenate the newly generated samples to the original data set
                ret_x = numpy.concatenate((self.x, sx1, sx2), axis=0)
                ret_y = numpy.concatenate((self.y, sy1, sy2), axis=0)

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

            # Boolean array with True for noisy support vectors
            noise_bool = []
            for x in support_vector:
                noise_bool.append(self.is_noise(x, self.y, self.minc,
                                                self.nearest_neighbour_))

            # Turn into array#
            noise_bool = asarray(noise_bool)

            # Remove noisy support vectors
            support_vector = support_vector[numpy.logical_not(noise_bool)]

            # Find support_vectors there are in danger (interpolation) or not
            # (extrapolation)
            danger_bool = [self.in_danger(x, self.y, self.m, self.minc,
                                          self.nearest_neighbour_)
                           for x in support_vector]

            # Turn into array#
            danger_bool = asarray(danger_bool)

            #Something ...#
            safety_bool = numpy.logical_not(danger_bool)

            #things to print#
            print_stats = (len(support_vector),
                           noise_bool.sum(),
                           danger_bool.sum(),
                           safety_bool.sum())

            if self.verbose:
                print("Out of %i support vectors, %i are noisy, %i are in danger "
                      "and %i are safe." % print_stats)

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
            numpy.random.seed(self.rs)
            fractions = betavariate(alpha=10, beta=10)


            # Interpolate samples in danger
            nns = self.nearest_neighbour_.kneighbors(support_vector[danger_bool],
                                                     return_distance=False)[:, 1:]

            sx1, sy1 = self.make_samples(support_vector[danger_bool], minx,
                                         self.minc, nns,
                                         fractions * (int(self.ratio * len(minx)) + 1),
                                         step_size=1,
                                         random_state=self.rs)

            # Extrapolate safe samples
            nns = self.nearest_neighbour_.kneighbors(support_vector[safety_bool],
                                                     return_distance=False)[:, 1:]

            sx2, sy2 = self.make_samples(support_vector[safety_bool], minx,
                                         self.minc, nns,
                                         (1 - fractions) * int(self.ratio * len(minx)),
                                         step_size=-self.out_step,
                                         random_state=self.rs)

            if self.verbose:
                print("done!")

            # Concatenate the newly generated samples to the original data set
            ret_x = concatenate((self.x, sx1, sx2), axis=0)
            ret_y = concatenate((self.y, sy1, sy2), axis=0)

            return ret_x, ret_y