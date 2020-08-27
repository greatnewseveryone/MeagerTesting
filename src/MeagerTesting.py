'''
Intended as a template for models that estimate conditional treatment effects in A/B tests

Based on Rachel Meager's credit program assessment paper.
All observations on a unit of measurement are handled as results of correlated treatments.
The full correlation is factored into correlations over treatment, time, and outcome variable,
allowing the pooling of signal across all available measurements.

Can be interpreted as a hierarchical multi-task Gaussian Process regression
similar to Bonilla et al and Lawrence et al

TODO:
- add conditioning on features of individuals
    - similar to the time embeddings can add another cov factor
      based on something like umap or learned coordinates as in gplvm
    - alternatively can use additive global terms
- to scale better and to omit params in missing conditions
  can learn a subset of param values and use covariances to get the full set

'''

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as fun

import pyro
import pyro.distributions as dist

from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.guides import AutoLowRankMultivariateNormal

from psis import psisloo

class GPCATE(object):
    '''

    '''

    def __init__(self,
                 lr= .01,
                 n_iter= 3000,
                 n_stp= 2,
                 notebook= True):

        self.notebook= notebook
        self.lr= lr
        self.n_iter= n_iter
        self.guide= AutoLowRankMultivariateNormal(self.model)
        self.lrn_crvs= []
        self.n_iter= n_iter
        self.n_stp= n_stp
        self.data= None
        self.posterior= []


    def get_distr(self, data, pars):
        '''
        Compute model predictions
        '''
        wts= (pars['glb'] + pars['ind']).view(data['n_ind'], data['n_trt'], data['n_tms'], data['n_mrk'])
        ab= torch.cat([wts[:,:1], wts[:,:1]+wts[:,1:]], dim=1)  # baseline + additive treatment effect
        t0= pars['t0'] + data['t0']                             # empirical bayes to anchor the mean outcome values at t0 of each individual
        evl= t0 + ab                                            # can add replicate and additional bias terms here
        nze= pars['noise'].exp().log1p() + data['noise']        # option to include measurement standard errors
        return dist.MultivariateNormal(evl[data['individuals'], data['treatments'], data['time_inds']],
                                       nze[data['individuals'], data['treatments'], data['time_inds']].unsqueeze(1) * torch.eye(data['n_mrk']))


    def model(self, data):
        '''
        Define the parameters
        '''
        n_ind= data['n_ind']
        n_trt= data['n_trt']
        n_tms= data['n_tms']
        n_mrk= data['n_mrk']

        n_prs= n_trt*n_tms*n_mrk

        plt_ind= pyro.plate('individuals', n_ind, dim=-3)
        plt_trt= pyro.plate('treatments', n_trt, dim=-2)
        plt_tms= pyro.plate('times', n_tms, dim=-1)

        pars= {}
        # covariance factors
        with plt_tms:
            pars['dt0']= pyro.sample('dt0', dist.Normal(0,1))
            pars['dt1']= pyro.sample('dt1', dist.Normal(0,1))

        pars['theta_trt0']= pyro.sample('theta_trt0', dist.HalfCauchy(torch.ones(n_trt)))
        pars['theta_mrk0']= pyro.sample('theta_mrk0', dist.HalfCauchy(torch.ones(n_mrk)))
        pars['theta_trt1']= pyro.sample('theta_trt1', dist.HalfCauchy(torch.ones(n_trt)))
        pars['L_omega_trt1']= pyro.sample('L_omega_trt1', dist.LKJCorrCholesky(n_trt, torch.ones(1)))
        pars['theta_mrk1']= pyro.sample('theta_mrk1', dist.HalfCauchy(torch.ones(n_mrk)))
        pars['L_omega_mrk1']= pyro.sample('L_omega_mrk1', dist.LKJCorrCholesky(n_mrk, torch.ones(1)))

        times0= fun.pad(torch.cumsum(pars['dt0'].exp().log1p(), 0), (1,0), value=0)[:-1].unsqueeze(1)
        times1= fun.pad(torch.cumsum(pars['dt1'].exp().log1p(), 0), (1,0), value=0)[:-1].unsqueeze(1)
        cov_t0= (-torch.cdist(times0, times0)).exp()
        cov_t1= (-torch.cdist(times1, times1)).exp()

        cov_i0= pars['theta_trt0'].diag()
        L_Omega_trt= torch.mm(torch.diag(pars['theta_trt1'].sqrt()), pars['L_omega_trt1'])
        cov_i1= L_Omega_trt.mm(L_Omega_trt.t())

        cov_m0= pars['theta_mrk0'].diag()
        L_Omega_mrk= torch.mm(torch.diag(pars['theta_mrk1'].sqrt()), pars['L_omega_mrk1'])
        cov_m1= L_Omega_mrk.mm(L_Omega_mrk.t())

        # kronecker product of the factors
        cov_itm0= torch.einsum('ij,tu,mn->itmjun',[cov_i0, cov_t0, cov_m0]).view(n_prs, n_prs)
        cov_itm1= torch.einsum('ij,tu,mn->itmjun',[cov_i1, cov_t1, cov_m1]).view(n_prs, n_prs)

        # global and individual level params of each marker, treatment, and time point
        pars['glb']= pyro.sample('glb', dist.MultivariateNormal(torch.zeros(n_prs), cov_itm0))
        with plt_ind:
            pars['ind']= pyro.sample('ind', dist.MultivariateNormal(torch.zeros(n_prs), cov_itm1))

        # observation noise, time series bias and scale
        pars['noise_scale']= pyro.sample('noise_scale', dist.HalfCauchy(torch.ones(n_mrk)))
        pars['t0_scale']= pyro.sample('t0_scale', dist.HalfCauchy(torch.ones(n_mrk)))
        with plt_ind:
            pars['t0']= pyro.sample('t0', dist.MultivariateNormal(torch.zeros(n_mrk),pars['t0_scale'].diag()))
            with plt_trt, plt_tms:
                pars['noise']= pyro.sample('noise', dist.MultivariateNormal(torch.zeros(n_mrk),pars['noise_scale'].diag()))

        # likelihood of the data
        distr= self.get_distr(data, pars)
        pyro.sample('obs', distr, obs=data['Y'])


    def get_posterior(self, n_samp= 500):
        with torch.no_grad():
            while len(self.posterior)<n_samp:
                self.posterior.append(self.guide())
        return self.posterior


    def get_cred_int(self, param= 'ab', n_samp= 500, quantiles= (.05,.5,.95)):
        data= self.data

        wts= torch.stack([i['glb']+i['ind'] for i in self.get_posterior(n_samp)])
        wts= wts.view(n_samp, data['n_ind'], data['n_trt'], data['n_tms'], data['n_mrk'])
        if param=='ab':
            vals= torch.cat([wts[:,:,:1], wts[:,:,:1]+wts[:,:,1:]], dim=2)
        elif param=='incr':
            vals= wts[:,:,1:]
        cred_int= torch.zeros(len(quantiles),
                              data['n_ind'],
                              data['n_trt'],
                              data['n_tms'],
                              data['n_mrk'])
        for ind in range(vals.shape[1]):
            for trt in range(int(param=='incr'), vals.shape[2]):
                for tm in range(vals.shape[3]):
                    for mrk in range(vals.shape[4]):
                        vals_srt= torch.sort(vals[:,ind,trt,tm,mrk])[0]
                        for qi,q in enumerate(quantiles):
                            cred_int[qi,ind,trt,tm,mrk]= vals_srt[int(n_samp*q)]
        return cred_int


    def get_external_validity(self):
        '''
        Model performance metrics

        psisloo reweights samples of pointwise posterior likelihoods to approximate likelihoods under leave one out cross validation
            ref: https://arxiv.org/pdf/1507.04544.pdf
        lmbd measures the extent to which parameters are pooled across individuals
            ref: http://www.stat.columbia.edu/~gelman/research/published/rsquared.pdf
        '''
        assert self.data is not None, 'train the model first'

        eps_smps= []
        llk_smps= []
        for pars in self.get_posterior():
            llk_smps.append(self.get_distr(self.data, pars).log_prob(self.data['Y']))
            eps_smps.append(pars['ind'])

        loo, loos, ks= psisloo(torch.stack(llk_smps).t().numpy())
        eps_smps= torch.stack(eps_smps)
        lmbd= 1 - (eps_smps.mean(0).var(0) / eps_smps.var(1).mean(0))
        return lmbd.squeeze(), loo, (ks<.7).mean()


    def get_data_dict(self, T, W, X, Y, S= 1e-7):
        data= { 'Y'           : torch.tensor(Y).float(),
                'noise'       : torch.tensor(S),
                'individuals' : torch.LongTensor(X),
                'treatments'  : torch.LongTensor(W)}

        T= list(T)
        time_inds= { t:i for i,t in enumerate(sorted(set(T)))}
        data['time_inds']= torch.LongTensor([time_inds[t] for t in T])
        data['time_vals']= torch.FloatTensor(sorted(time_inds.keys())).unsqueeze(1)

        data['n_tms']= int(len(time_inds))
        data['n_ind']= int(max(X) + 1)
        data['n_trt']= int(max(W) + 1)
        data['n_mrk']= int(Y.shape[1])

        t0= []
        for i in range(data['individuals'].max()+1):
            t0.append(data['Y'][(data['individuals']==i) & (data['time_inds']==0)].mean(0))
        data['t0']= torch.stack(t0).unsqueeze(1).unsqueeze(1)

        self.data= data
        return data


    def fit(self, T, W, X, Y, S= 1e-7):
        data= self.get_data_dict(T, W, X, Y, S)

        lr= self.lr
        svi = SVI(self.model, self.guide, Adam({'lr': lr}), loss=Trace_ELBO())

        lc= []
        lt= []
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        if self.notebook: from tqdm.notebook import tqdm
        else: from tqdm import tqdm

        for i in tqdm(range(self.n_iter)):
            elbo= svi.step(data)
            lt.append(elbo)

            if i and not i%(self.n_iter//self.n_stp):
                lr*=.1
                svi = SVI(self.model, self.guide, Adam({'lr': lr}), loss=Trace_ELBO())

            if not i%(self.n_iter//20):
                with torch.no_grad():
                    lc.append(sum(lt)/len(lt))
                    lt= []
                    pars= self.guide()
                    distr= self.get_distr(data, pars)
                    llk= distr.log_prob(data['Y']).mean().item()
                    r2= np.corrcoef(distr.mean.view(-1), data['Y'].view(-1))[0,1]**2
                    print('%d\t\tELBO: %.2E - LLK: %.2E - r2: %.3f - lr: %.2E'%(i, elbo, llk, r2, lr))
                    self.lrn_crvs.append((i, elbo, llk, r2, lr))
