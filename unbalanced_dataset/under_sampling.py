from unbalanced_dataset.unbalanced_dataset import *
__author__ = 'fmfnogueira'


class UnderSampler(UnbalancedDataset):
    """
    Object to under sample the majority class(es) by randomly picking samples
    with replacement.
    """

    def __init__(self, ratio=1., random_state=None):
        """
        :param ratio:
            The ratio of majority elements to sample with respect to the number
            of minority cases.

        :param random_state:
            Seed.

        :return:
            underx, undery: The features and target values of the under-sampled
            data set.
        """

        # Passes the relevant parameters back to the parent class.
        UnbalancedDataset.__init__(self, ratio=ratio, random_state=random_state)

    def resample(self):
        """
        ...
        """

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

        # Loop over the other classes under picking at random
        for key in self.ucd.keys():
            # If the minority class is up, skip it
            if key == self.minc:
                continue

            # Set the ratio to be no more than the number of samples available
            if self.ratio * self.ucd[self.minc] > self.ucd[key]:
                num_samples = self.ucd[key]
            else:
                num_samples = int(self.ratio * self.ucd[self.minc])

            # Pick some elements at random
            numpy.random.seed(self.rs)
            indx = numpy.random.randint(low=0, high=self.ucd[key],
                                        size=num_samples)

            # Concatenate to the minority class
            underx = concatenate((underx, self.x[self.y == key][indx]), axis=0)
            undery = concatenate((undery, self.y[self.y == key][indx]), axis=0)

        return underx, undery


class TomekLinks(UnbalancedDataset):
    """
    Object to identify and remove majority samples that form a Tomek link with
    minority samples.
    """

    def __init__(self):
        """
        No parameters.

        :return:
            Nothing.
        """

        UnbalancedDataset.__init__(self)

    def resample(self):
        """
        :return:
            Return the data with majority samples that form a Tomek link
            removed.
        """

        from sklearn.neighbors import NearestNeighbors

        # Find the nearest neighbour of every point
        print("Finding nearest neighbour...", end="")
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(self.x)
        nns = nn.kneighbors(self.x, return_distance=False)[:, 1]
        print("done!")

        # Send the information to is_tomek function to get boolean vector back
        print("Looking for majority Tomek links...", end="")
        links = self.is_tomek(self.y, nns, self.minc)

        # Return data set without majority Tomek links.
        return self.x[numpy.logical_not(links)], self.y[numpy.logical_not(links)]


class ClusterCentroids(UnbalancedDataset):
    """
    Experimental method that under samples the majority class by replacing a
    cluster of majority samples by the cluster centroid of a KMeans algorithm.

    This algorithm keeps N majority samples by fitting the KMeans algorithm
    with N cluster to the majority class and using the coordinates of the N
    cluster centroids as the new majority samples.
    """

    def __init__(self, ratio=1, random_state=None, **kwargs):
        """
        :param kwargs:
            Arguments the user might want to pass to the KMeans object from
            scikit-learn.

        :param ratio:
            The number of cluster to fit with respect to the number of samples
            in the minority class.
            N_clusters = int(ratio * N_minority_samples) = N_maj_undersampled.

        :param random_state:
            Seed.

        :return:
            Under sampled data set.
        """
        UnbalancedDataset.__init__(self, ratio=ratio,
                                   random_state=random_state)

        self.kwargs = kwargs

    def resample(self):
        """
        ???

        :return:
        """

        # Create the clustering object
        from sklearn.cluster import KMeans
        kmeans = KMeans(random_state=self.rs)
        kmeans.set_params(**self.kwargs)

        # Start with the minority class
        underx = self.x[self.y == self.minc]
        undery = self.y[self.y == self.minc]

        # Loop over the other classes under picking at random
        print('Finding cluster centroids...', end="")
        for key in self.ucd.keys():
            # If the minority class is up, skip it.
            if key == self.minc:
                continue

            # Set the number of clusters to be no more than the number of
            # samples
            if self.ratio * self.ucd[self.minc] > self.ucd[key]:
                n_clusters = self.ucd[key]
            else:
                n_clusters = int(self.ratio * self.ucd[self.minc])

            # Set the number of clusters and find the centroids
            kmeans.set_params(n_clusters=n_clusters)
            kmeans.fit(self.x[self.y == key])
            centroids = kmeans.cluster_centers_

            # Concatenate to the minority class
            underx = concatenate((underx, centroids), axis=0)
            undery = concatenate((undery, ones(n_clusters) * key), axis=0)
            print(".", end="")

        print("done!")

        return underx, undery
