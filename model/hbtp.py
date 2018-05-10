import numpy as np
import time
from scipy.special import gammaln, psi
from collections import defaultdict
from corpus import BaseCorpus
from model import BaseModel
from RBFKernel import RBFKernel
from copy import deepcopy
from sklearn.cluster import KMeans
eps = 1e-100


class Corpus(BaseCorpus):

    def __init__(self, vocab, word_ids, word_cnt, child_to_parent_and_story, story_to_users, n_topic, length_scale=1.0, noise_precision=10.0, lrate=0.001): # 0.0001, (0.01 doesn't work)
        super().__init__(vocab, word_ids, word_cnt, n_topic)

        self.story_to_users = story_to_users

        self.n_user = len(child_to_parent_and_story)
        self.n_edge = sum([len(edges) for edges in child_to_parent_and_story.values()])
        self.rootid = len(child_to_parent_and_story)

        # Note that the size of A and B is [self.n_edge, self.n_topic], not [self.M, self.n_topic]
        self.A = np.random.gamma(shape=1, scale=1, size=[self.n_edge, self.n_topic])
        self.B = np.random.gamma(shape=1, scale=1, size=[self.n_edge, self.n_topic])
        self.lnZ_edge = psi(self.A) - np.log(self.B)
        self.Z_edge = self.A / self.B

        user_edgerows = defaultdict(list)
        edgerow_story = list()
        edgerow_parent = list()
        cnt = 0

        # (child:int, [(parent:int, story:int), ...])
        for child, parent_and_story in child_to_parent_and_story.items():
            for parent, story in parent_and_story:
                user_edgerows[child].append(cnt)
                edgerow_story.append(story)
                edgerow_parent.append(parent)
                cnt += 1

        # np.array(list of int)
        self.edgerow_story = np.array(edgerow_story)
        self.edgerow_parent = np.array(edgerow_parent)

        # GP-LVM settings
        self.kernel = RBFKernel(length_scale)
        # self.h = np.random.randint(20, 40, self.M)
        self.h = np.random.rand(self.M) + 2
        self.P = 25

        self.c1 = np.zeros((self.M, self.n_topic))
        self.mu_y = np.zeros((self.P)) # corresponds to M in GPSTM code
        self.Sigma_y = np.zeros((self.P, self.P)) # corresponds to S in GPSTM code

        self.lrate = lrate
        self.noise_precision = noise_precision
        self.InpNoise = np.ones([2, 2]) * np.sqrt(0.1)

        # {user:int -> edgerows:int}
        self.user_edgerows = dict(user_edgerows)

        # Saving C
        self.C = dict()

    def safe_inv(self, matrix_):
        return np.linalg.inv(matrix_ + np.identity(matrix_.shape[0]) * 1e-10)


class HBTP(BaseModel):
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

    def __init__(self, n_topic, n_voca, alpha=5., beta=5., dir_prior=0.5):
        super().__init__(n_topic, n_voca, alpha, beta, dir_prior)
        self.GP_iter = 10

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

        for iteration in range(max_iter):
            lb = 0
            curr = time.clock()
            lb += self.update_C(corpus, False, iteration)
            lb += self.update_Z(corpus)
            lb += self.update_V(corpus)
            if (iteration + 1) % 5 == 0:
                self.update_GPLV(corpus, iteration)
            print('%d iter, %.2f time, %.2f lower_bound' % (iteration, time.clock() - curr, lb))

            if iteration > 3:
                self.lbs.append(lb)
            """
            if iteration > 5:
                if (abs(self.lbs[-1] - self.lbs[-2]) / abs(self.lbs[-2])) < 1e-5:
                    break
                if self.lbs[-1] < self.lbs[-2]:
                    break
            """

    # update per word v.d. phi
    def update_C(self, corpus, is_heldout, thisiter):
        corpus.phi_doc = np.zeros([corpus.M, self.n_topic])
        psiGamma = psi(self.gamma)
        gammaSum = np.sum(self.gamma, 0)
        psiGammaSum = psi(np.sum(self.gamma, 0))

        lnZ_user = np.zeros([corpus.n_user, corpus.n_topic])
        Z_user = np.zeros([corpus.n_user, corpus.n_topic])

        for i in range(corpus.n_user):
            lnZ_user[i] = np.mean(corpus.lnZ_edge[corpus.user_edgerows[i]], axis=0)
            Z_user[i] = np.mean(corpus.Z_edge[corpus.user_edgerows[i]], axis=0)

        lnZ = np.zeros([corpus.M, corpus.n_topic])
        Z = np.zeros([corpus.M, corpus.n_topic])

        for i in range(corpus.M):
            lnZ[i] = np.mean(lnZ_user[corpus.story_to_users[i]], axis=0)
            Z[i] = np.mean(Z_user[corpus.story_to_users[i]], axis=0)

        lb = 0
        if self.is_compute_lb:
            # expectation of p(eta) over variational q(eta)
            l1 = self.n_topic * gammaln(self.dir_prior * self.n_voca) \
                 - self.n_topic * self.n_voca * gammaln(self.dir_prior) \
                 - np.sum((self.dir_prior - 1) * (psiGamma - psiGammaSum))
            lb += l1
            # entropy of q(eta)
            l2 = np.sum(gammaln(gammaSum)) - np.sum(gammaln(self.gamma)) \
                 + np.sum((self.gamma - 1) * (psiGamma - psiGammaSum))
            lb -= l2

        if not is_heldout:
            # multinomial topic distribution prior
            self.gamma = np.zeros([self.n_voca, self.n_topic]) + self.dir_prior

        for m in range(corpus.M):
            ids = corpus.word_ids[m]
            cnt = corpus.word_cnt[m]
            Nm = np.sum(cnt)

            # C = len(ids) x K
            E_ln_eta = psiGamma[ids, :] - psiGammaSum
            if thisiter < 5:
                C = np.exp(E_ln_eta + lnZ[m, :])
            else:
                C = np.exp(E_ln_eta + lnZ[m, :] \
                    + (corpus.c1[m] / np.float(Nm) - (corpus.phi_doc[m] - corpus.C[m]  + 1)/ (2 * np.float(Nm)**2 )) * corpus.noise_precision)
            C = C / np.sum(C, 1)[:, np.newaxis]
            corpus.C[m] = C

            if not is_heldout:
                self.gamma[ids, :] += cnt[:, np.newaxis] * C
            corpus.phi_doc[m, :] = np.sum(cnt[:, np.newaxis] * C, 0)

            if self.is_compute_lb:
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

        Z_user = np.zeros([corpus.n_user, corpus.n_topic])

        for i in range(corpus.n_user):
            Z_user[i] = np.mean(corpus.Z_edge[corpus.user_edgerows[i]], axis=0)

        p_user = Z_user / np.sum(Z_user, axis=1)[:, np.newaxis]
        p_user = np.vstack((p_user, self.p))
        H = corpus.h[corpus.edgerow_story]
        H[corpus.edgerow_parent == corpus.rootid] = 0.
        bph = self.beta * p_user[corpus.edgerow_parent] * np.exp(H[:, np.newaxis])

        xi = np.sum(corpus.A / corpus.B, 1)  # m dim
        corpus.A = bph + corpus.phi_doc[corpus.edgerow_story]
        corpus.B = 1 + (corpus.Nm[corpus.edgerow_story] / xi)[:, np.newaxis]
        corpus.lnZ_edge = psi(corpus.A) - np.log(corpus.B)
        corpus.Z_edge = corpus.A / corpus.B

        if self.is_compute_lb:
            # expectation of p(Z)
            corpus.lnZ_edge = psi(corpus.A) - np.log(corpus.B)
            l1 = np.sum((bph - 1) * corpus.lnZ_edge) - np.sum(corpus.A / corpus.B) - np.sum(gammaln(bph))
            lb += l1
            # entropy of q(Z)
            l2 = np.sum(corpus.A * np.log(corpus.B)) + np.sum((corpus.A - 1) * corpus.lnZ_edge) \
                 - np.sum(corpus.A) - np.sum(gammaln(corpus.A))
            lb -= l2

        return lb

    # coordinate ascent for V
    def update_V(self, corpus):
        lb = 0

        sumLnZ = np.sum(corpus.lnZ_edge[corpus.edgerow_parent == corpus.rootid], 0)  # K dim
        n_edges_with_root_parents = sum(corpus.edgerow_parent == corpus.rootid)

        for i in range(self.c_a_max_step):
            one_V = 1 - self.V
            stickLeft = self.getStickLeft(self.V)  # prod(1-V_(dim-1))
            p = self.V * stickLeft

            psiV = psi(self.beta * p)

            vVec = self.beta * stickLeft * sumLnZ - n_edges_with_root_parents * self.beta * stickLeft * psiV

            for k in range(self.n_topic):
                tmp2 = self.beta * sum(sumLnZ[k + 1:] * p[k + 1:] / one_V[k])
                tmp3 = n_edges_with_root_parents * self.beta * sum(psiV[k + 1:] * p[k + 1:] / one_V[k])
                vVec[k] = vVec[k] - tmp2
                vVec[k] = vVec[k] + tmp3
                vVec[k] = vVec[k]
            vVec[:self.n_topic - 2] -= (self.alpha - 1) / one_V[:self.n_topic - 2]
            vVec[self.n_topic - 1] = 0
            step_stick = self.getstepSTICK(self.V, vVec, sumLnZ, self.beta, self.alpha, n_edges_with_root_parents)
            self.V = self.V + step_stick * vVec
            print(step_stick)
            self.p = self.getP(self.V)

        if self.is_compute_lb:
            # expectation of p(V)
            lb += (self.n_topic - 1) * gammaln(self.alpha + 1) \
                  - (self.n_topic - 1) * gammaln(self.alpha) \
                  + np.sum((self.alpha - 1) * np.log(1 - self.V[:-1]))

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
        if np.sum(step_zero > 0) > 0:
            min_zero = min(step_zero[step_zero > 0])
        if np.sum(step_one > 0) > 0:
            min_one = min(step_one[step_one > 0])
        max_step = min([min_zero, min_one])

        if max_step > 0:
            step_check_vec = np.array([0., .01, .125, .25, .375, .5, .625, .75, .875]) * max_step
        else:
            step_check_vec = list()

        f = np.zeros(len(step_check_vec))
        for ite in range(len(step_check_vec)):
            step_check = step_check_vec[ite]
            vec_check = curr + step_check * grad
            p = self.getP(vec_check)
            f[ite] = - M * np.sum(gammaln(beta * p)) \
                     + np.sum((beta * p - 1) * sumlnZ) + (alpha - 1.) * np.sum(np.log(1. - vec_check[:-1] + eps))

        if len(f) != 0:
            b = f.argsort()[-1]
            step = step_check_vec[b]
        else:
            step = 0

        if b == 1:
            rho = .5
            keep_cont = True
            fold = f[b]
            while keep_cont:
                step = rho * step
                vec_check = curr + step * grad
                tmp = np.zeros(vec_check.size)
                tmp[1:] = vec_check[:-1]
                p = vec_check * np.cumprod(1 - tmp)
                fnew = - M * np.sum(gammaln(beta * p)) \
                       + np.sum((beta * p - 1) * sumlnZ) + (alpha - 1.) * np.sum(np.log(1. - vec_check[:-1] + eps))
                if fnew > fold:
                    fold = fnew
                else:
                    keep_cont = False
            step = step / rho
        return step

    def update_GPLV(self, corpus, thisiter):
        doc_means = corpus.phi_doc / np.sum(corpus.phi_doc, axis=1)[:, np.newaxis]
        c1 = deepcopy(doc_means)

        # for _ in range(self.GP_iter):

        #     kmmodel = KMeans(n_clusters=corpus.P, n_init=10, init='random')
        #     # kmmodel.fit(np.float(C[Ytr[:, rr] == 1, :]))
        #     kmmodel.fit(c1[corpus.h > np.mean(corpus.h)])
        #     inducing_points = kmmodel.cluster_centers_
        #     Kgg = corpus.kernel.selfCompute(inducing_points)
        #     Kgg_inv = corpus.safe_inv(Kgg)
        #     # Kgg_inv = np.linalg.inv(Kgg)
        #     # (mu_y, Sigma_y, EKgc, EKgcKgcT) = self.update_GP(Kgg_inv, inducing_points, C, Ytr, corpus.InpNoise)

        #     EKgc = corpus.kernel.EVzx(inducing_points, c1, corpus.InpNoise)
        #     EKgcKgcT = np.sum(corpus.kernel.EVzxVzxT(inducing_points, c1, corpus.InpNoise), axis=0)

        #     Sigma_y = corpus.safe_inv(Kgg_inv + corpus.noise_precision * Kgg_inv.dot(EKgcKgcT).dot(Kgg_inv))
        #     # Sigma_y = np.linalg.inv(Kgg_inv + corpus.noise_precision * Kgg_inv.dot(EKgcKgcT).dot(Kgg_inv))
        #     mu_y = corpus.noise_precision * Sigma_y.dot(Kgg_inv).dot(EKgc).dot(corpus.h)
        #     if thisiter == 4:
        #         gC = -corpus.noise_precision * c1 + corpus.noise_precision * doc_means
        #     else:
        #         gC = -corpus.noise_precision * corpus.c1 + corpus.noise_precision * doc_means

        #     for kk in range(corpus.n_topic):
        #         grad_EKcg = corpus.kernel.grad_EVzx_by_mu_batch(EKgc, inducing_points, c1, corpus.InpNoise, kk)
        #         grad_EKgcgcT_tensor = corpus.kernel.grad_EVzxVzxT_by_mu_batch(EKgcKgcT, inducing_points, c1, corpus.InpNoise, kk)
        #         Multiplier = Kgg_inv.dot(mu_y.dot(mu_y.T) + Sigma_y).dot(Kgg_inv) - Kgg_inv
        #         Term2 = np.zeros([corpus.M, ])
        #         for dd in range(corpus.M):
        #             Term2[dd] = grad_EKgcgcT_tensor[dd, :, :].dot(Multiplier).trace()

        #         gC[:, kk] += -0.5 * np.float(corpus.noise_precision) * Term2

        #         label_mat = np.tile(corpus.h, [corpus.P, 1]).T
        #         gC[:, kk] += np.float64(corpus.noise_precision) * (label_mat * grad_EKcg).dot(Kgg_inv).dot(mu_y).ravel()

        #     c1 = c1 + corpus.lrate * gC
        #     print(np.mean(np.abs(gC)))

        EgC_square = np.zeros((corpus.M, corpus.n_topic))

        for _ in range(self.GP_iter):

            kmmodel = KMeans(n_clusters=corpus.P, n_init=10, init='random')
            # kmmodel.fit(np.float(C[Ytr[:, rr] == 1, :]))
            kmmodel.fit(c1[corpus.h > np.mean(corpus.h)])
            inducing_points = kmmodel.cluster_centers_
            Kgg = corpus.kernel.selfCompute(inducing_points)
            Kgg_inv = corpus.safe_inv(Kgg)
            # Kgg_inv = np.linalg.inv(Kgg)
            # (mu_y, Sigma_y, EKgc, EKgcKgcT) = self.update_GP(Kgg_inv, inducing_points, C, Ytr, corpus.InpNoise)

            EKgc = corpus.kernel.EVzx(inducing_points, c1, corpus.InpNoise)
            EKgcKgcT = np.sum(corpus.kernel.EVzxVzxT(inducing_points, c1, corpus.InpNoise), axis=0)

            Sigma_y = corpus.safe_inv(Kgg_inv + corpus.noise_precision * Kgg_inv.dot(EKgcKgcT).dot(Kgg_inv))
            # Sigma_y = np.linalg.inv(Kgg_inv + corpus.noise_precision * Kgg_inv.dot(EKgcKgcT).dot(Kgg_inv))
            mu_y = corpus.noise_precision * Sigma_y.dot(Kgg_inv).dot(EKgc).dot(corpus.h)
            if thisiter == 4:
                gC = -corpus.noise_precision * c1 + corpus.noise_precision * doc_means
            else:
                gC = -corpus.noise_precision * corpus.c1 + corpus.noise_precision * doc_means

            for kk in range(corpus.n_topic):
                grad_EKcg = corpus.kernel.grad_EVzx_by_mu_batch(EKgc, inducing_points, c1, corpus.InpNoise, kk)
                grad_EKgcgcT_tensor = corpus.kernel.grad_EVzxVzxT_by_mu_batch(EKgcKgcT, inducing_points, c1, corpus.InpNoise, kk)
                Multiplier = Kgg_inv.dot(mu_y.dot(mu_y.T) + Sigma_y).dot(Kgg_inv) - Kgg_inv
                Term2 = np.zeros([corpus.M, ])
                for dd in range(corpus.M):
                    Term2[dd] = grad_EKgcgcT_tensor[dd, :, :].dot(Multiplier).trace()

                gC[:, kk] += -0.5 * np.float(corpus.noise_precision) * Term2

                label_mat = np.tile(corpus.h, [corpus.P, 1]).T
                gC[:, kk] += np.float64(corpus.noise_precision) * (label_mat * grad_EKcg).dot(Kgg_inv).dot(mu_y).ravel()

            EgC_square = 0.9 * EgC_square + 0.1 * np.power(gC, 2)
            c1 = c1 + corpus.lrate / np.sqrt( EgC_square + 1e-8 ) * gC
            print(np.mean(np.abs(gC)), np.mean(np.abs(corpus.lrate / np.sqrt( EgC_square + 1e-8 ))))

        corpus.c1 = c1
        self.mu_y = mu_y
        self.Sigma_y = Sigma_y
        self.inducing_points = inducing_points
        self.Kgg_inv = Kgg_inv
        self.Kgg = Kgg
