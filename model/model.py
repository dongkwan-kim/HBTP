import numpy as np
import os
import pickle


class BaseModel:

    def __init__(self, n_topic, n_voca, alpha=5., beta=5., dir_prior=0.5):
        """
        Basic TopicModel class for HDP, DLIN, HBTP.
        Attributes
        ----------
        n_topic: int
            number of truncated topics for variational inference
        n_voca: int
            vocabulary size
        """

        # Hyper-parameters: 5., 5., 0.5
        self.alpha = alpha
        self.beta = beta
        self.dir_prior = dir_prior

        self.n_topic = n_topic
        self.n_voca = n_voca
        self.V = np.zeros(self.n_topic)

        # for even p
        self.V[0] = 1. / self.n_topic
        for k in range(1, n_topic - 1):
            self.V[k] = (1. / self.n_topic) / np.prod(1. - self.V[:k])
        self.V[self.n_topic - 1] = 1.

        self.p = self.getP(self.V)
        self.mean = np.zeros(self.n_topic)
        self.gamma = np.random.gamma(shape=1, scale=1, size=[self.n_voca, self.n_topic]) + self.dir_prior
        self.c_a_max_step = 5
        self.is_compute_lb = False
        self.lbs = []

    def fit(self, corpus, max_iter=100):
        """ Run variational EM to fit the model"""
        raise NotImplementedError

    def getStickLeft(self, V):
        stl = np.ones(self.n_topic)
        stl[1:] = np.cumprod(1. - V)[:-1]
        return stl

    def getP(self, V):
        one_v = np.ones(self.n_topic)
        one_v[1:] = (1. - V)[:-1]
        p = V * np.cumprod(one_v)
        return p

    def write_top_words(self, corpus, filepath):
        with open(filepath, 'w') as f:
            for ti in range(corpus.K):
                top_words = corpus.vocab[self.gamma[:, ti].argsort()[::-1][:20]]
                f.write('%d,%f' % (ti, self.p[ti]))
                for word in top_words:
                    f.write(',' + word)
                f.write('\n')

    def save_result(self, folder, corpus):
        if not os.path.exists(folder):
            os.mkdir(folder)
        np.savetxt(folder + '/final_mu.csv', corpus.mu, delimiter=',')
        np.savetxt(folder + '/final_sigma.csv', corpus.sigma, delimiter=',')
        np.savetxt(folder + '/final_mean.csv', self.mean, delimiter=',')
        np.savetxt(folder + '/final_V.csv', self.V, delimiter=',')
        self.write_top_words(corpus, folder + '/final_top_words.csv')
        pickle.dump(self, open(folder + '/model.pkl', 'w'))
