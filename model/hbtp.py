import numpy as np
import time
from scipy.special import gammaln, psi
from scipy.stats import spearmanr
from collections import defaultdict
from corpus import BaseCorpus
from model import BaseModel
from RBFKernel import RBFKernel
from copy import deepcopy
from sklearn.cluster import KMeans

eps = 1e-100


def safe_inv(matrix_):
    return np.linalg.inv(matrix_ + np.identity(matrix_.shape[0]) * 1e-10)


class Corpus(BaseCorpus):

    def __init__(self, vocab, word_ids, word_cnt, child_to_parent_and_story, story_to_users, n_topic):
        super().__init__(vocab, word_ids, word_cnt, n_topic)

        self.story_to_users = story_to_users

        self.n_user = len(child_to_parent_and_story)
        self.n_edge = sum([len(edges) for edges in child_to_parent_and_story.values()])
        self.rootid = len(child_to_parent_and_story)

        # Note that the size of A and B is [self.n_edge, n_topic], not [self.M, n_topic]
        self.A = np.random.gamma(shape=1, scale=1, size=[self.n_edge, n_topic])
        self.B = np.random.gamma(shape=1, scale=1, size=[self.n_edge, n_topic])
        self.lnZ_edge = psi(self.A) - np.log(self.B + eps)
        self.Z_edge = self.A / self.B

        user_edgerows = defaultdict(list)
        edgerow_story = list()
        edgerow_parent = list()
        story_edgerow = defaultdict(list)
        story_parent = defaultdict(list)

        cnt = 0
        # (child:int, [(parent:int, story:int), ...])
        for child, parent_and_story in child_to_parent_and_story.items():
            for parent, story in parent_and_story:
                user_edgerows[child].append(cnt)
                edgerow_story.append(story)
                edgerow_parent.append(parent)
                story_edgerow[story].append(cnt)
                story_parent[story].append(parent)
                cnt += 1

        # np.array(list of int)
        self.edgerow_story = np.array(edgerow_story)
        self.edgerow_parent = np.array(edgerow_parent)

        # {user:int -> edgerows:int}
        self.user_edgerows = dict(user_edgerows)
        self.story_edgerow = dict(story_edgerow)
        self.story_parent = dict(story_parent)

        # GP-LVM settings
        self.h = np.random.rand(self.M)
        self.h_original = deepcopy(self.h)

        self.c1 = np.zeros((self.M, n_topic))

        # Saving C
        self.C = dict()

        # Z_user
        self.lnZ_user = np.zeros([self.n_user, n_topic])
        self.Z_user = np.zeros([self.n_user, n_topic])

        for i in range(self.n_user):
            self.lnZ_user[i] = np.mean(self.lnZ_edge[self.user_edgerows[i]], axis=0)
            self.Z_user[i] = np.mean(self.Z_edge[self.user_edgerows[i]], axis=0)


class HBTP(BaseModel):
    """
    Homogeneity-Based Transmissive Process (HBTP)
    Jooyeon Kim, Dongkwan Kim, Alice Oh, 2019
    The 12th ACM International Conference on Web Search and Data Mining (WSDM)
    Attributes
    ----------
    n_topic: int
        number of truncated topics for variational inference
    n_voca: int
        vocabulary size
    """

    def __init__(self, n_topic, n_voca, alpha=5., beta=10., dir_prior=1e-2):
        super().__init__(n_topic, n_voca, alpha, beta, dir_prior)

        length_scale = 0.1
        self.kernel = RBFKernel(length_scale)

        # Noise precision for updating c1
        noise_precision = 0.1
        self.noise_precision = noise_precision

        self.GP_update_every = 5

    def fit(self, corpus, max_iter=300):
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
            curr = time.clock()
            self.update_C(corpus, iteration)
            self.update_Z(corpus)
            self.update_V(corpus)
            self.update_alpha_and_beta(corpus)
            if (iteration + 1) % self.GP_update_every == 0:
                self.update_GPLV(corpus)
                self.update_hindex(corpus)
            print('%d iter, %.2f time' % (iteration, time.clock() - curr))

    # update per word v.d. phi
    def update_C(self, corpus, thisiter):
        corpus.phi_doc = np.zeros([corpus.M, self.n_topic])
        psiGamma = psi(self.gamma)
        gammaSum = np.sum(self.gamma, 0)
        psiGammaSum = psi(np.sum(self.gamma, 0))

        lnZ = np.zeros([corpus.M, self.n_topic])
        Z = np.zeros([corpus.M, self.n_topic])

        for i in range(corpus.M):
            lnZ[i] = np.mean(corpus.lnZ_user[corpus.story_to_users[i]], axis=0)
            Z[i] = np.mean(corpus.Z_user[corpus.story_to_users[i]], axis=0)

        self.gamma = np.zeros([self.n_voca, self.n_topic]) + self.dir_prior

        for m in range(corpus.M):
            ids = corpus.word_ids[m]
            cnt = corpus.word_cnt[m]
            Nm = np.sum(cnt)

            E_ln_eta = psiGamma[ids, :] - psiGammaSum
            if thisiter < self.GP_update_every:
                C = np.exp(E_ln_eta + lnZ[m, :])
            else:
                C = np.exp(E_ln_eta + lnZ[m, :] \
                           + (corpus.c1[m] / np.float(Nm) - (corpus.phi_doc[m] - corpus.C[m] + 1) / (
                            2 * np.float(Nm) ** 2)) * self.noise_precision)
            # C = np.exp(E_ln_eta + lnZ[m, :])

            C = C / (np.sum(C, 1) + eps)[:, np.newaxis]
            corpus.C[m] = C

            self.gamma[ids, :] += cnt[:, np.newaxis] * C
            corpus.phi_doc[m, :] = np.sum(cnt[:, np.newaxis] * C, 0)

    # update variational gamma prior a and b for Z_mk
    def update_Z(self, corpus):
        p_user = corpus.Z_user / (np.sum(corpus.Z_user, axis=1) + eps)[:, np.newaxis]
        p_user = np.vstack((p_user, self.p))
        H = corpus.h[corpus.edgerow_story]
        H[corpus.edgerow_parent == corpus.rootid] = 0.
        bph = self.beta * p_user[corpus.edgerow_parent] * np.exp(H[:, np.newaxis])

        xi = np.sum(corpus.A / (corpus.B + eps), 1)  # m dim
        corpus.A = bph + corpus.phi_doc[corpus.edgerow_story]
        corpus.B = 1 + (corpus.Nm[corpus.edgerow_story] / (xi + eps))[:, np.newaxis]
        corpus.lnZ_edge = psi(corpus.A) - np.log(corpus.B + eps)
        corpus.Z_edge = corpus.A / (corpus.B + eps)

        corpus.lnZ_user = np.zeros([corpus.n_user, self.n_topic])
        corpus.Z_user = np.zeros([corpus.n_user, self.n_topic])

        for i in range(corpus.n_user):
            corpus.lnZ_user[i] = np.mean(corpus.lnZ_edge[corpus.user_edgerows[i]], axis=0)
            corpus.Z_user[i] = np.mean(corpus.Z_edge[corpus.user_edgerows[i]], axis=0)

    # coordinate ascent for V
    def update_V(self, corpus):
        sumLnZ = np.sum(corpus.lnZ_edge[corpus.edgerow_parent == corpus.rootid], 0)  # K dim
        n_edges_with_root_parents = sum(corpus.edgerow_parent == corpus.rootid)

        for i in range(self.c_a_max_step):
            one_V = 1 - self.V
            stickLeft = self.getStickLeft(self.V)  # prod(1-V_(dim-1))
            p = self.V * stickLeft

            psiV = psi(self.beta * p)

            vVec = self.beta * stickLeft * sumLnZ - n_edges_with_root_parents * self.beta * stickLeft * psiV

            for k in range(self.n_topic):
                tmp2 = self.beta * sum(sumLnZ[k + 1:] * p[k + 1:] / (one_V[k] + eps))
                tmp3 = n_edges_with_root_parents * self.beta * sum(psiV[k + 1:] * p[k + 1:] / (one_V[k] + eps))
                vVec[k] = vVec[k] - tmp2
                vVec[k] = vVec[k] + tmp3
                vVec[k] = vVec[k]
            vVec[:self.n_topic - 2] -= (self.alpha - 1) / (one_V[:self.n_topic - 2] + eps)
            vVec[self.n_topic - 1] = 0
            step_stick = self.getstepSTICK(self.V, vVec, sumLnZ, self.beta, self.alpha, n_edges_with_root_parents)
            self.V = self.V + step_stick * vVec
            self.p = self.getP(self.V)

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

    def update_GPLV(self, corpus):
        # Noise for the inducing points
        InpNoise = np.ones([2, 2]) * 0.0001

        GP_iter = 20
        lrate_gC = 0.001
        P = 20
        doc_means = corpus.phi_doc / (np.sum(corpus.phi_doc, axis=1) + eps)[:, np.newaxis]
        if np.sum(corpus.c1) != 0:
            c1 = corpus.c1
        else:
            c1 = deepcopy(doc_means)

        EgC_square = np.zeros((corpus.M, self.n_topic))

        for _ in range(GP_iter):

            kmmodel = KMeans(n_clusters=P, n_init=10, init='random')
            kmmodel.fit(c1[corpus.h > np.mean(corpus.h)])
            inducing_points = kmmodel.cluster_centers_
            Kgg = self.kernel.selfCompute(inducing_points)
            Kgg_inv = safe_inv(Kgg)

            EKgc = self.kernel.EVzx(inducing_points, c1, InpNoise)
            EKgcKgcT = np.sum(self.kernel.EVzxVzxT(inducing_points, c1, InpNoise), axis=0)

            Sigma_y = safe_inv(Kgg_inv + self.noise_precision * Kgg_inv.dot(EKgcKgcT).dot(Kgg_inv))
            mu_y = self.noise_precision * Sigma_y.dot(Kgg_inv).dot(EKgc).dot(corpus.h)
            gC = -self.noise_precision * c1 + self.noise_precision * doc_means

            for kk in range(self.n_topic):
                grad_EKcg = self.kernel.grad_EVzx_by_mu_batch(EKgc, inducing_points, c1, InpNoise, kk)
                grad_EKgcgcT_tensor = self.kernel.grad_EVzxVzxT_by_mu_batch(EKgcKgcT, inducing_points, c1, InpNoise, kk)
                Multiplier = Kgg_inv.dot(mu_y.dot(mu_y.T) + Sigma_y).dot(Kgg_inv) - Kgg_inv
                Term2 = np.zeros([corpus.M, ])
                for dd in range(corpus.M):
                    Term2[dd] = grad_EKgcgcT_tensor[dd, :, :].dot(Multiplier).trace()

                gC[:, kk] += -0.5 * np.float(self.noise_precision) * Term2

                label_mat = np.tile(corpus.h, [P, 1]).T
                gC[:, kk] += np.float64(self.noise_precision) * (label_mat * grad_EKcg).dot(Kgg_inv).dot(mu_y).ravel()

            corpus.c1 = c1
            corpus.inducing_points = inducing_points
            corpus.Kgg = Kgg

            EgC_square = 0.9 * EgC_square + 0.1 * np.power(gC, 2)
            c1 = c1 + lrate_gC / np.sqrt(EgC_square + 1e-8) * gC
            print('gc', np.mean(np.abs(gC)), 'lrate for RMSProp: ',
                  np.mean(np.abs(lrate_gC / np.sqrt(EgC_square + 1e-8))), 'spearmanr: w.r.t. initialized doc_means: ',
                  np.mean([spearmanr(c1[ii], doc_means[ii]) for ii in range(corpus.M)]))

    def update_hindex(self, corpus):
        h_iter = 100
        lrate_gh = 0.001
        xi_inv = 0.1
        kappa = 10.

        psi_1 = np.prod(
            np.exp(-0.5 * np.power(corpus.c1[:, np.newaxis, :] - corpus.inducing_points[np.newaxis, :, :], 2) \
                   / (xi_inv + 1)) * np.power(xi_inv + 1, -0.5), axis=2)

        first_term = -0.25 * np.power(
            corpus.inducing_points[np.newaxis, :, np.newaxis, :] - corpus.inducing_points[np.newaxis, np.newaxis, :, :],
            2)
        inducing_bar = (corpus.inducing_points[np.newaxis, :, np.newaxis, :] + corpus.inducing_points[np.newaxis,
                                                                               np.newaxis, :, :]) / 2
        second_term = np.power(corpus.c1[:, np.newaxis, np.newaxis] - inducing_bar, 2) / (2 * xi_inv + 1)

        psi_2_s = np.prod(np.exp(first_term - second_term) * np.power(2 * xi_inv + 1, -0.5), axis=3)
        psi_2 = np.sum(psi_2_s, axis=0)

        tmp = safe_inv(kappa * psi_2 + corpus.Kgg)
        tmp2 = np.dot(psi_1, tmp)
        tmp3 = np.dot(tmp2, psi_1.T)
        W = kappa * np.identity(corpus.M) - kappa ** 2 * tmp3

        p_user = corpus.Z_user / (np.sum(corpus.Z_user, axis=1) + eps)[:, np.newaxis]
        p_user = np.vstack((p_user, self.p))

        Egh_square = np.zeros(corpus.M)
        for _ in range(h_iter):
            gh = np.zeros(corpus.M)
            for mm in range(corpus.M):
                bph = self.beta * p_user[corpus.story_parent[mm]] * np.exp(corpus.h[mm])
                tmp4 = np.sum(bph * (corpus.lnZ_edge[corpus.story_edgerow[mm]] - psi(bph))) + np.sum(
                    (W[mm, :] + W[:, mm]) * corpus.h)
                gh[mm] = tmp4

            Egh_square = 0.9 * Egh_square + 0.1 * np.power(gh, 2)
            corpus.h = corpus.h + lrate_gh / np.sqrt(Egh_square + 1e-8) * gh
            print('h mean', np.mean(corpus.h), 'gh', np.mean(np.abs(gh)), 'lrate for RMSProp: ',
                  np.mean(np.abs(lrate_gh / np.sqrt(Egh_square + 1e-8))), 'spearmanr w.r.t. initialized h-index: ',
                  spearmanr(corpus.h, corpus.h_original))

    def update_alpha_and_beta(self, corpus):
        b_iter = 1000
        minibatch_size = 1000
        lrate_gb = 0.001
        tau_1 = 1.
        tau_2 = 1e-3
        self.alpha = (self.n_topic + tau_1 - 2) / (tau_2 - np.sum(np.log(1 - self.V[:-1])) + eps)

        p_user = corpus.Z_user / (np.sum(corpus.Z_user, axis=1) + eps)[:, np.newaxis]
        p_user = np.vstack((p_user, self.p))
        H = corpus.h[corpus.edgerow_story]
        H[corpus.edgerow_parent == corpus.rootid] = 0.
        ph = p_user[corpus.edgerow_parent] * np.exp(H[:, np.newaxis])

        kappa_1 = 1.
        kappa_2 = 1e-3
        Egb_square = 0.
        for _ in range(b_iter):
            minibatch = np.random.choice(np.arange(len(corpus.lnZ_edge)), size=minibatch_size)
            # gb = np.sum(ph * (corpus.lnZ_edge - psi(self.beta * ph)) ) + (kappa_1 - 1) / (self.beta + eps) - kappa_2            
            gb = np.sum(ph[minibatch] * (corpus.lnZ_edge[minibatch] - psi(self.beta * ph[minibatch]))) * len(
                corpus.lnZ_edge) / minibatch_size + (kappa_1 - 1) / (self.beta + eps) - kappa_2
            Egb_square = 0.9 * Egb_square + 0.1 * np.power(gb, 2)
            self.beta = self.beta + lrate_gb / np.sqrt(Egb_square + 1e-8) * gb
            # print('alpha: ', self.alpha, 'beta: ', self.beta, 'gb', gb, 'lrate for RMSProp: ', np.mean(np.abs(lrate_gb / np.sqrt( Egb_square + 1e-8 ))))
        print('alpha: ', self.alpha, 'beta: ', self.beta)
