import numpy as np
import time
from scipy.special import gammaln, psi
from corpus import BaseCorpus
from collections import defaultdict


eps = 1e-100


class Corpus(BaseCorpus):

    def __init__(self, vocab, word_ids, word_cnt, child_to_parent_and_story, story_to_users, n_topic):
        super().__init__(vocab, word_ids, word_cnt, n_topic)

        self.story_to_users = story_to_users

        self.n_user = len(child_to_parent_and_story)
        self.n_edge = sum([len(edges) for edges in child_to_parent_and_story.values()])

        # Note that the size of A and B is [self.n_edge, self.n_topic], not [self.M, self.n_topic]
        self.A = np.random.gamma(shape=1, scale=1, size=[self.n_edge, self.n_topic])
        self.B = np.random.gamma(shape=1, scale=1, size=[self.n_edge, self.n_topic])
        self.lnZ_edge = psi(self.A) - np.log(self.B)
        self.Z_edge = self.A / self.B

        user_edgerows = defaultdict(list)
        edgerow_story = list()
        cnt = 0

        # (child:int, [(parent:int, story:int), ...])
        for child, parent_and_story in child_to_parent_and_story.items():
            for parent, story in parent_and_story:
                user_edgerows[child].append(cnt)
                edgerow_story.append(story)
                cnt += 1

        # np.array(list of int)
        self.edgerow_story = np.array(edgerow_story)

        # {user:int -> edgerows:int}
        self.user_edgerows = dict(user_edgerows)


class HBTP:
    """
    Homogeneity-Based Transmissive Process (HBTP)
    Jooyeon Kim, Dongkwan Kim, Alice Oh, 2018
    Attributes
    ----------
    n_topic: int
        number of truncated topics for variational inference
    n_voca: int
        vocabulary size
    """

    def __init__(self, n_topic, n_voca):
        self.n_topic = n_topic
        self.n_voca = n_voca  # vocabulary size
        self.V = np.zeros(self.n_topic)

        # for even p
        self.V[0] = 1. / self.n_topic
        for k in range(1, n_topic - 1):
            self.V[k] = (1. / self.n_topic) / np.prod(1. - self.V[:k])
        self.V[self.n_topic - 1] = 1.

        self.p = self.getP(self.V)
        self.alpha = 5.
        self.beta = 5.
        self.dir_prior = 0.5
        self.mean = np.zeros(self.n_topic)
        self.gamma = np.random.gamma(shape=1, scale=1, size=[self.n_voca, self.n_topic]) + self.dir_prior
        self.c_a_max_step = 5
        self.is_compute_lb = True
        self.lbs = []

    def fit(self, corpus, max_iter=100):
        """ Run variational EM to fit the model
        Parameters
        ----------
        max_iter: int
            maximum number of iterations
        corpus:
        Returns
        -------
        """

        for iter in range(max_iter):
            lb = 0
            curr = time.clock()
            lb += self.update_C(corpus, False)
            lb += self.update_Z(corpus)
            lb += self.update_V(corpus)
            print('%d iter, %.2f time, %.2f lower_bound' % (iter, time.clock() - curr, lb))

            if iter > 3:
                self.lbs.append(lb)
                if iter > 5:
                    if (abs(self.lbs[-1] - self.lbs[-2]) / abs(self.lbs[-2])) < 1e-5:
                        break
                    if (self.lbs[-1] < self.lbs[-2]):
                        break

    def getStickLeft(self, V):
        stl = np.ones(self.n_topic)
        stl[1:] = np.cumprod(1. - V)[:-1]
        return stl

    def getP(self, V):
        one_v = np.ones(self.n_topic)
        one_v[1:] = (1. - V)[:-1]
        p = V * np.cumprod(one_v)
        return p

    # update per word v.d. phi
    def update_C(self, corpus, is_heldout):

        corpus.phi_doc = np.zeros([corpus.M, self.n_topic])
        psiGamma = psi(self.gamma)
        gammaSum = np.sum(self.gamma, 0)
        psiGammaSum = psi(np.sum(self.gamma, 0))

        lnZ_user = np.zeros([corpus.n_user, corpus.n_topic])
        Z_user = np.zeros([corpus.n_user, corpus.n_topic])

        for i in range(corpus.n_user):
            lnZ_user[i] =  np.mean(corpus.lnZ_edge[corpus.user_edgerows[i]], axis = 0)
            Z_user[i] =  np.mean(corpus.Z_edge[corpus.user_edgerows[i]], axis = 0)

        lnZ = np.zeros([corpus.M, corpus.n_topic])
        Z = np.zeros([corpus.M, corpus.n_topic])

        for i in range(corpus.M):
            lnZ[i] =  np.mean(lnZ_user[corpus.story_to_users[i]], axis = 0)
            Z[i] =  np.mean(Z_user[corpus.story_to_users[i]], axis = 0)

        lb = 0
        if (self.is_compute_lb):
            # expectation of p(eta) over variational q(eta)
            l1 = self.n_topic * gammaln(self.dir_prior * self.n_voca) - self.n_topic * self.n_voca * gammaln(self.dir_prior) - np.sum(
                (self.dir_prior - 1) * (psiGamma - psiGammaSum))
            lb += l1
            # entropy of q(eta)
            l2 = np.sum(gammaln(gammaSum)) - np.sum(gammaln(self.gamma)) + np.sum(
                (self.gamma - 1) * (psiGamma - psiGammaSum))
            lb -= l2

        if not is_heldout:
            self.gamma = np.zeros([self.n_voca, self.n_topic]) + self.dir_prior  # multinomial topic distribution prior

        for m in range(corpus.M):
            ids = corpus.word_ids[m]
            cnt = corpus.word_cnt[m]

            # C = len(ids) x K
            E_ln_eta = psiGamma[ids, :] - psiGammaSum
            C = np.exp(E_ln_eta + lnZ[m, :])
            C = C / np.sum(C, 1)[:, np.newaxis]

            if not is_heldout:
                self.gamma[ids, :] += cnt[:, np.newaxis] * C
            corpus.phi_doc[m, :] = np.sum(cnt[:, np.newaxis] * C, 0)

            if (self.is_compute_lb):
                # expectation of p(X) over variational q
                lb += np.sum(cnt[:, np.newaxis] * C * E_ln_eta)
                # expectation of p(C) over variational q
                l1 = np.sum(cnt[:, np.newaxis] * C * (lnZ[m, :] - np.log(np.sum(Z[m, :]))))
                lb += l1
                # entropy of q(C)
                l2 = np.sum(cnt[:, np.newaxis] * C * np.log(C + eps))
                lb -= l2

        # print ' E[p(eta,C,X)]-E[q(eta,C)] = %f' % lb

        return lb

    # update variational gamma prior a and b for Z_mk
    def update_Z(self, corpus):
        lb = 0
        bp = self.beta * self.p

        xi = np.sum(corpus.A / corpus.B, 1)  # m dim
        corpus.A = bp + corpus.phi_doc[corpus.edgerow_story]
        corpus.B = 1 + (corpus.Nm[corpus.edgerow_story] / xi)[:, np.newaxis]
        corpus.lnZ_edge = psi(corpus.A) - np.log(corpus.B)
        corpus.Z_edge = corpus.A / corpus.B

        if (self.is_compute_lb):
            # expectation of p(Z)
            E_ln_Z = psi(corpus.A) - np.log(corpus.B)
            l1 = np.sum((bp - 1) * (E_ln_Z)) - np.sum(
                corpus.A / corpus.B) - corpus.n_edge * np.sum(gammaln(bp))
            lb += l1
            # entropy of q(Z)
            l2 = np.sum(corpus.A * np.log(corpus.B)) + np.sum((corpus.A - 1) * (E_ln_Z)) - np.sum(corpus.A) - np.sum(
                gammaln(corpus.A))
            lb -= l2
            # print ' E[p(Z)]-E[q(Z)] = %f' % lb

        return lb

    # coordinate ascent for V
    def update_V(self, corpus):
        lb = 0

        sumLnZ = np.sum(corpus.lnZ_edge, 0) # K dim
        for i in range(self.c_a_max_step):
            one_V = 1 - self.V
            stickLeft = self.getStickLeft(self.V)  # prod(1-V_(dim-1))
            p = self.V * stickLeft

            psiV = psi(self.beta * p)

            vVec = self.beta * stickLeft * sumLnZ - corpus.n_edge * self.beta * stickLeft * psiV;

            for k in range(self.n_topic):
                tmp2 = self.beta * sum(sumLnZ[k + 1:] * p[k + 1:] / one_V[k]);
                tmp3 = corpus.n_edge * self.beta * sum(psiV[k + 1:] * p[k + 1:] / one_V[k]);
                vVec[k] = vVec[k] - tmp2;
                vVec[k] = vVec[k] + tmp3;
                vVec[k] = vVec[k]
            vVec[:self.n_topic - 2] -= (self.alpha - 1) / one_V[:self.n_topic - 2];
            vVec[self.n_topic - 1] = 0;
            step_stick = self.getstepSTICK(self.V, vVec, sumLnZ, self.beta, self.alpha, corpus.n_edge);
            self.V = self.V + step_stick * vVec;
            self.p = self.getP(self.V)

        if self.is_compute_lb:
            # expectation of p(V)
            lb += (self.n_topic - 1) * gammaln(self.alpha + 1) - (self.n_topic - 1) * gammaln(self.alpha) + np.sum(
                (self.alpha - 1) * np.log(1 - self.V[:-1]))
            # print ' E[p(V)]-E[q(V)] = %f' % lb

        # print '%f diff     %f' % (new_ll - old_ll, lb)
        return lb

    # get stick length to update the gradient
    def getstepSTICK(self, curr, grad, sumlnZ, beta, alpha, M):
        _curr = curr[:len(curr) - 1]
        _grad = grad[:len(curr) - 1]
        _curr = _curr[_grad != 0]
        _grad = _grad[_grad != 0]

        step_zero = -_curr / _grad
        step_one = (1 - _curr) / _grad
        min_zero = 1
        min_one = 1
        if (np.sum(step_zero > 0) > 0):
            min_zero = min(step_zero[step_zero > 0])
        if (np.sum(step_one > 0) > 0):
            min_one = min(step_one[step_one > 0])
        max_step = min([min_zero, min_one]);

        if max_step > 0:
            step_check_vec = np.array([0., .01, .125, .25, .375, .5, .625, .75, .875]) * max_step;
        else:
            step_check_vec = list();

        f = np.zeros(len(step_check_vec));
        for ite in range(len(step_check_vec)):
            step_check = step_check_vec[ite];
            vec_check = curr + step_check * grad;
            p = self.getP(vec_check)
            f[ite] =  - M * np.sum(gammaln(beta * p)) + np.sum((beta * p - 1) * sumlnZ)\
                     + (alpha - 1.) * np.sum(np.log(1. - vec_check[:-1] + eps))

        if len(f) != 0:
            b = f.argsort()[-1]
            step = step_check_vec[b]
        else:
            step = 0;

        if b == 1:
            rho = .5;
            bool = 1;
            fold = f[b];
            while bool:
                step = rho * step;
                vec_check = curr + step * grad;
                tmp = np.zeros(vec_check.size)
                tmp[1:] = vec_check[:-1]
                p = vec_check * np.cumprod(1 - tmp)
                fnew =  - M * np.sum(gammaln(beta * p)) + np.sum((beta * p - 1) * sumlnZ) \
                       + (alpha - 1.) * np.sum(np.log(1. - vec_check[:-1] + eps))
                if fnew > fold:
                    fold = fnew
                else:
                    bool = 0
            step = step / rho
        return step

    def write_top_words(self, corpus, filepath):
        with open(filepath, 'w') as f:
            for ti in range(corpus.K):
                top_words = corpus.vocab[self.gamma[:, ti].argsort()[::-1][:20]]
                f.write('%d,%f' % (ti, self.p[ti]))
                for word in top_words:
                    f.write(',' + word)
                f.write('\n')

    def save_result(self, folder, corpus):
        import os, pickle
        if not os.path.exists(folder):
            os.mkdir(folder)
        np.savetxt(folder + '/final_V.csv', self.V, delimiter=',')
        self.write_top_words(corpus, folder + '/final_top_words.csv')
        self.write_corr_topics(corpus, folder + '/final_corr_topics.csv')
        pickle.dump(self, open(folder + '/model.pkl', 'w'))