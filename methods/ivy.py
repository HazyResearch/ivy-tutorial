import numpy as np
from utils.constant import NEGATIVE, POSITIVE
from utils.utils import mean_to_cannonical
import networkx as nx
from collections import Counter
from sklearn.linear_model import LinearRegression, LogisticRegression
import random
from utils.utils import numberToBase
from utils.utils import multi_index_to_index
from utils.utils import sigmoid, outliers_iqr, threshold_matrix
from utils.r_pca  import R_pca

class Ivy:

    def __init__(self):
        self.valid_iv = None
        self.w = None

    def train(self,IVs,X=None, deps=[], anchor=None, class_balance=None, use_forward=False, use_canonical=False, **kwargs):

        # determine valid IVs
        if self.valid_iv == None:
            self.valid_iv = list(range(IVs.shape[1]))
        # self._is_valid_iv(IVs)

        # record use_forward and use_canonical
        self.use_forward = use_forward
        self.use_canonical = use_canonical

        # change the encoding of IVs to {-1,1}
        IVs = self._convert_iv(IVs)

        # assign class balance
        if class_balance is None:
            class_balance = Counter(IVs.flatten())
            class_balance = class_balance[1]/(class_balance[-1]+class_balance[1])
            class_balance = [1-class_balance, class_balance]

        n,p = IVs.shape
        #  sem: second empirical moment matirx
        sem = np.matmul(IVs.T,IVs)/n

        # for conditional independent model, 
        # the inverse graph is the complete graph
        inverse_graph = nx.complete_graph(p)

        # incidence matrix of the inverse graph
        M = nx.incidence_matrix(inverse_graph).T.toarray()
        edge_list_M = np.asarray(inverse_graph.edges())


        if self.use_forward and (deps == []):

            # create q
            exp_q = np.abs(sem[edge_list_M[:,0],edge_list_M[:,1]])
            # handle zeros in exp_q
            eps_zero = 1e-12
            exp_q[exp_q==0] = eps_zero
            # take log of exp_q to get q
            q = np.log(exp_q)

            # use only positive entries in sem
            selector_positive_sem = sem[edge_list_M[:,0],edge_list_M[:,1]]>0
            q_positve = q[selector_positive_sem]
            M_postive = M[selector_positive_sem,:]

            # make sure that M matrix is full-rank
            # find the all-zero column
            selector_non_zero_column_M_positve = np.sum(M_postive,axis=0)>0

            # compute l
            # r_cond = None is for using future default 
            # and silence warning
            l_subset,_,_,_ = np.linalg.lstsq(
                M_postive[:,selector_non_zero_column_M_positve],
                q_positve,rcond=None)
            l = np.zeros(M.shape[1])
            l[:] = np.nan
            l[selector_non_zero_column_M_positve] = l_subset
            l[np.isnan(l)] = -100000

            self.w = np.exp(l)

        else: # with dependency
    
            # did not really use forward algorithm despite specified in the input
            self.use_forward = False

            # make sure deps are in increasing order
            deps = [[int(x[0]),int(x[1])] for x in deps]

            # update M and edge_list_M by removing edges in deps
            selector = [(list(x) not in deps) for x in edge_list_M]
            M = M[selector,:]
            edge_list_M = edge_list_M[selector,:]

            # make sure that M matrix is full-rank
            # find the all-zero column
            selector_non_zero_column_M = np.sum(M,axis=0)>0

            # compute q
            # {0,1} encoding of the (inverse) covariance matrix
            pinv_sem = np.linalg.pinv(np.cov((IVs.T+1)/2))
            exp_q = pinv_sem[edge_list_M[:,0],edge_list_M[:,1]]
            q =  np.log(np.abs(exp_q))

            # # determine if the z share the same sign or not
            # # not completely finished 
            # sign = np.sign(pinv_sem)
            # np.fill_diagonal(sign,0)
            # same_sign = np.vstack(np.where(sign<0)).T
            # same_sign = same_sign[same_sign[:,0]<same_sign[:,1],:]
            # same_sign_graph = nx.Graph()
            # same_sign_graph.add_edges_from(same_sign)
            # [print(c) for c in nx.connected_components(same_sign_graph)]

            # compute l and w (mu)
            l,_,_,_ = np.linalg.lstsq(M,q,rcond=None)

            # abs_z
            abs_z = np.exp(l)

            # did not decide the sign of z
            # one workaround that can consider is to pick only 
            # the negative entries of the inverse matrix when estimating
            # using least square, not sure why though
            
            # also, Sigma_O @ z = np.sqrt(c) * Sigma_OS
            # see if Sigma_OS>0 as a check?
            z = abs_z

        # get the w which is the mean parameter
        # the following are in {0,1} encoding
        self.class_balance = class_balance #[neg, pos]
        var_cb  = self.class_balance[1] - self.class_balance[1]**2
        mean_est = np.mean((IVs+1)/2,axis=0).reshape(-1,1)
        cov_est_obs = np.cov((IVs.T+1)/2)

        if not self.use_forward:
            c_hat = 1.0 / var_cb * (1 + z.T @ cov_est_obs @ z)
            Sigma_OS_hat = cov_est_obs @ z / np.sqrt(c_hat)

            # force mu to be non-negative
            # second moment in {0,1} encoding
            self.w = (Sigma_OS_hat + self.class_balance[1]*mean_est.T).flatten()

            # map to {-1,1} encoding to represent accuracy (difference)
            self.w = 4*self.w - 2*mean_est.T - 2*self.class_balance[1] + 1
            self.w = self.w[0]

        # conditional accuracy
        mean_est_aug = np.append((mean_est),self.class_balance[1])
        B = np.array([[1,1,0,0],[1,0,1,0],[1,0,0,1],[1,1,1,1]])
        self.conditional_accuracy = None

        # P(x=1,z=1) P(x=1,z=-1) P(x=-1,z=1) P(x=-1,z=-1)
        for i in range(p):
            b = np.array([mean_est_aug[i],mean_est_aug[p],(self.w[i]+1)/2,1])
            beta,_,_,_ = np.linalg.lstsq(B,b,rcond=None)
            if self.conditional_accuracy is None:
                self.conditional_accuracy = np.array([
                    beta[3]/self.class_balance[0],
                    beta[0]/self.class_balance[1]])
            else:  
                self.conditional_accuracy = np.vstack([
                    self.conditional_accuracy, np.array([
                    beta[3]/self.class_balance[0],
                    beta[0]/self.class_balance[1]])
                ])

        # compute auc
        tn = self.conditional_accuracy[:,0]*self.class_balance[0]
        tp = self.conditional_accuracy[:,1]*self.class_balance[1]
        fn = self.class_balance[1]-tp
        fp = self.class_balance[0]-tn
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        auc = tpr*fpr/2 + (tpr+1)*(1-fpr)/2
        self.auc = auc   

        # # pick conditional independent LFs from cliques
        # # create dependency graph
        # g=nx.Graph()
        # g.add_nodes_from(range(len(self.w)))
        # g.add_edges_from(deps)

        # # pick a representative from a clique
        # selector = [list(curr)[np.argmax(self.w[list(curr)])] for curr in nx.connected_components(g)]
        # self.w[list(set(range(len(self.w))).difference(selector))] = -1
        
        # # # average accuracy over clique
        # # for curr in nx.connected_components(g):
        # #     self.w[list(curr)] = self.w[list(curr)]/len(list(curr))

        if self.use_canonical:

            # first empirical moments with {-1,1} encoding
            fem = np.mean(IVs,axis=0)
            # second empirical moments with {-1,1} encoding
            sem = ((IVs.T) @ (IVs))/(n)
            # augmented with z
            fem_aug = np.append(fem,self.class_balance[1]-self.class_balance[0])
            sem_aug = np.vstack([sem,self.w])
            sem_aug = np.hstack([sem_aug,np.append(self.w,0).reshape(-1,1)])
            mean_parameter = np.copy(sem_aug)
            np.fill_diagonal(mean_parameter,fem_aug)

            self.im, self.theta = mean_to_cannonical(deps,mean_parameter,maxiter=100,alpha=0.2)

            # update class balance
            self.class_balance_im = self.im.infer.query([str(p)],show_progress=False).normalize(inplace=False).values

            # use cannonical parameters
            self.im.infer.calibrate()
            self.im.infer.get_cliques()
            factor_dict = self.im.infer.get_clique_beliefs()

            # create prob_dict
            prob_dict = {}
            for clique in factor_dict:
                prob_index = np.array([numberToBase(x,3,len(clique)) for x in range(3**len(clique))])
                prob = []
                factor = factor_dict[clique].normalize(inplace=False)
                for assignment in prob_index:
                    # see if assignment is the same dimension of factor
                    absence_selector = (assignment==1)
                    if np.sum(absence_selector) == 0:
                        prob.append(factor.values[tuple((assignment/2).astype(int))])
                    else:
                        # if not, marginalize:
                        margin_factor = factor.marginalize(np.array(factor.scope())[np.where(absence_selector)],inplace=False)
                        prob.append(margin_factor.values[tuple((assignment[~absence_selector]/2).astype(int))])
                # compute condtional probability upon z
                selector = (prob_index[:,factor.scope().index(str(p))]==0)
                prob = np.array(prob)
                prob[selector] = prob[selector]/class_balance[0]
                selector = (prob_index[:,factor.scope().index(str(p))]==2)
                prob[selector] = prob[selector]/class_balance[1]
                prob_dict.update({clique:{"prob_index":np.copy(prob_index),"prob":np.log(prob)}})

            self.prob_dict = prob_dict
    
    def dependency_validity(self,IVs,lam=None,mu=None,**kwargs):
        
        # take symmetric matrix as an argument as well
        if (IVs.shape[0]==IVs.shape[1]) and np.all(IVs==IVs.T):
            Sigma = IVs
        else:
           Sigma = np.cov(IVs.T)
        # compute the inverse covariance matrix
        pinv_Sigma = np.linalg.pinv(Sigma)

        # set default value of lam and mu
        lam_default = 1/np.sqrt(np.max(pinv_Sigma.shape))
        mu_default = np.prod(pinv_Sigma.shape)/(4*np.sum(np.abs(pinv_Sigma)))

        if lam == None:
            lam  = lam_default
        else:
            lam  = lam * lam_default
        if mu == None:
            mu = mu_default
        else:
            mu = mu * mu_default

        rpca = R_pca(pinv_Sigma,lmbda=lam,mu=mu)
        L, S = rpca.fit(max_iter=10000, iter_print=10000)

        # estimate accuracy
        u, s, vh = np.linalg.svd(L)
        l = -u[:,0]*np.sqrt(s[0])        
        score = (Sigma @ l)

        return score, S, L

    def predict_proba(self,IVs, is_ad_hoc=False):

        # compute soft label
        IVs = self._convert_iv(IVs)
        w = self.w
        conditional_accuracy = self.conditional_accuracy

        n,p = IVs.shape

        if self.use_forward:
            # use regular accuracy
            w = np.clip(np.abs(w),0.05,0.95)
            W =  (np.log(1+w)-np.log(1-w))
            
            Zprob = sigmoid(1/2* IVs @ W)
            Zprob_median = np.median(Zprob)

            Zprob_max = 0.9
            Zprob_min = 1-Zprob_max

            if Zprob_median>Zprob_max:
                Zprob_intercept = (np.log(Zprob_median)-np.log(1-Zprob_median)) - (np.log(Zprob_max)-np.log(1-Zprob_max))
                Zprob = sigmoid((1/2* IVs @ W)-Zprob_intercept)
            elif Zprob_median<Zprob_min:
                Zprob_intercept = (np.log(Zprob_min)-np.log(1-Zprob_min)) - (np.log(Zprob_median)-np.log(1-Zprob_median))
                Zprob = sigmoid((1/2* IVs @ W) + Zprob_intercept)
        else:
            # use conditional accuracy
            conditional_accuracy = np.clip(conditional_accuracy,
            np.min(np.abs(self.conditional_accuracy)),1)

            IVs_aug = np.hstack([
                (IVs==-1).astype(int),
                (IVs==1).astype(int)])

            conditional_accuracy_aug = np.vstack(
                # P(w_j=-1 \mid z=-1), P(w_j=1 \mid z=-1)
                [np.hstack([conditional_accuracy[:,0],1-conditional_accuracy[:,0]]),
                # P(w_j=-1 \mid z=1), P(w_j=1 \mid z=1)
                np.hstack([1-conditional_accuracy[:,1],conditional_accuracy[:,1]])]).T

            log_Zprob = IVs_aug @ np.log(conditional_accuracy_aug)

            Zprob = sigmoid(log_Zprob[:,1]-log_Zprob[:,0] + 
                np.log(self.class_balance[1])-np.log(self.class_balance[0]))

            if self.use_canonical:
                # use cannonical parameters
                pos_posterior = np.zeros(n)
                neg_posterior = np.zeros(n)
                for clique in self.prob_dict:
                    iv_index_in_clique = [x for x in range(len(clique)) if list(clique)[x]!= str(p)]
                    iv_index_in_IVs = np.array(list(clique))[iv_index_in_clique].astype(int)
                    assignment = np.zeros([n,len(clique)]).astype(int)
                    assignment[:,iv_index_in_clique] = IVs[:,iv_index_in_IVs]+1
                    pos_assignment = np.copy(assignment)
                    neg_assignment = np.copy(assignment)
                    pos_assignment[:,list(clique).index(str(p))] = 2
                    neg_assignment[:,list(clique).index(str(p))] = 0
                    del assignment
                    # find the index of the assignment
                    pos_assignment_single_index = multi_index_to_index(pos_assignment)
                    neg_assignment_single_index = multi_index_to_index(neg_assignment)
                    # update positve posterior
                    pos_posterior = pos_posterior + self.prob_dict[clique]["prob"][pos_assignment_single_index]
                    # update negative posterior
                    neg_posterior = neg_posterior + self.prob_dict[clique]["prob"][neg_assignment_single_index]

                Zprob = sigmoid(-(neg_posterior+np.log(self.class_balance_im[0]))+(pos_posterior+np.log(self.class_balance_im[1])))

        if is_ad_hoc:
            # w = self.auc
            # Zprob = np.matmul(IVs+1, w)/(IVs.shape[1]*2)

            # the first one underneath is the default criterion
            Zprob = np.matmul(IVs+1, w+1)/(IVs.shape[1]*2)

            # # use probability as synthesized IV
            # pass

            # # use log probability as synthesized IV
            # Zprob = 1/2* IVs @ (np.log(1+w) - np.log(1-w))/IVs.shape[1]
            
            # exp accuracy difference weight
            # Zprob = np.matmul(IVs, np.exp(w))/(IVs.shape[1]*2)

        return Zprob

    def predict(self, IVs, b=0.5):
        
        Zprob = self.predict_proba(IVs)
        Z = np.where(Zprob > np.median(Zprob), POSITIVE, NEGATIVE)
        return Z

    def _convert_iv(self,IVs):

        IVs = IVs-1
        IVs = IVs[:,self.valid_iv]
        return IVs

    def get_weights(self):
        return self.w

    def get_dependencies(self,S):

        # threshod deps_mat
        deps_mat = np.abs(np.copy(S))
        np.fill_diagonal(deps_mat,0)

        # compute thresh
        thresh_list = np.unique(np.abs(deps_mat[np.triu_indices(deps_mat.shape[0],k=1)]))
        outlier_index = outliers_iqr(thresh_list)[0]
        if len(outlier_index)>0:
            min_outlier = np.min(thresh_list[outlier_index])
            thresh = np.max(thresh_list[thresh_list<min_outlier])
            short_thresh_list = thresh_list[thresh_list>=thresh]
            thresh = short_thresh_list[np.argmax(short_thresh_list[1:]/short_thresh_list[0:-1])+1]
            # get the edges
            deps = threshold_matrix(deps_mat,thresh)
        else:
            deps = []

        return deps

    def get_valid_iv_indx(self,score):

        # get valid IVs
        valid_thresh_list = np.sort(np.abs(score))
        # exclue the first one
        valid_thresh = valid_thresh_list[np.argmax(valid_thresh_list[2:]/valid_thresh_list[1:-1])+2]
        valid_indx = np.where(np.abs(score)>=valid_thresh)[0]

        return valid_indx