import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
import scipy.stats as scs

class Propensity:
    '''
    Instance of propensity model to be used in a PU classification model
    '''
    def __init__(self):
        self.params = None
    
    def __str__(self):
        return str(self.params)
    
    def __repr__(self):
        return self.__str__()
    
    def initialization(self, Xe):
        self.params = np.random.rand(Xe.shape[1] + 1)
    
    def e(self, Xe):
        return 0.5*np.ones(Xe.shape[0])
    
    def loge(self, Xe):
        return np.zeros(Xe.shape[0])
    
    def predict_proba(self, Xe):
        return self.e(Xe)

    def predict_logproba(self, Xe):
        return self.loge(Xe)
    
    def predict(self, Xe, threshold=0.5):
        return (self.predict_proba(Xe)>=threshold).astype(int)

class LogProbitSig(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights):
        self.weights = weights
        super(LogProbitSig, self).__init__(endog, exog)

    def loge(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.norm.logcdf(np.log(pred), scale=np.exp(params[-1]))

    def log1e(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.norm.logsf(np.log(pred), scale=np.exp(params[-1]))
    
    def loglike(self, params):
        endog = self.endog
        return (self.weights*endog*self.loge(params) + (1-endog)*self.weights*self.log1e(params)).sum()
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.append('sig')
        if start_params is None:
            # Reasonable starting values
            # start_params = np.zeros((self.exog.shape[1] + 1))
            start_params = -0.5 + np.random.rand(self.exog.shape[1] + 1)
        return super(LogProbitSig, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)
class Gumbel(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights):
        self.weights = weights
        super(Gumbel, self).__init__(endog, exog)

    def loge(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.weibull_min.logcdf(pred, np.exp(params[-1]))

    def log1e(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.weibull_min.logsf(pred, np.exp(params[-1]))
    
    def loglike(self, params):
        endog = self.endog
        return (self.weights*endog*self.loge(params) + (1-endog)*self.weights*self.log1e(params)).sum()
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.append('b')
        if start_params is None:
            # Reasonable starting values
            # start_params = np.zeros((self.exog.shape[1] + 1))
            start_params = -0.5 + np.random.rand(self.exog.shape[1] + 1)
        return super(Gumbel, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)

class GaussianSN(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights):
        self.weights = weights
        super(GaussianSN, self).__init__(endog, exog)

    def loge(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.norm.logcdf(np.log(pred), scale=np.exp(params[-1]))

    def log1e(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.norm.logsf(np.log(pred), scale=np.exp(params[-1]))
    
    def logp(self, params):
        pred = np.dot(self.exog, np.exp(params[:-1]))
        return scs.norm.logpdf(np.log(pred), scale=np.exp(params[-1]))        

    def loglike(self, params):
        endog = self.endog
        return (self.weights*endog*self.logp(params) + (1-endog)*self.weights*self.log1e(params)).sum()

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.append('sig')
        if start_params is None:
            # Reasonable starting values
            # start_params = np.zeros((self.exog.shape[1] + 1))
            start_params = -0.5 + np.random.rand(self.exog.shape[1] + 1)
        return super(GaussianSN, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)


class LogProbitPropensity(Propensity):
    def __init__(self):
        return super(LogProbitPropensity, self).__init__()

    def __str__(self):
        if self.params is not None:
            s = 'Phi = {}'.format(self.params[:-1])
            s += '\n'
            s += 'sigma = {}'.format(self.params[-1])
            return s
        else:
            return ''

    def __repr__(self):
        return self.__str__()

    def initialization(self, Xe, gamma, Y, w=1.):
        self.params = np.random.rand(Xe.shape[1] + 1)
    
    def e(self, Xe):
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.max(np.concatenate([np.dot(Xe, self.params[:-1])[:,np.newaxis], 1e-30*np.zeros((Xe.shape[0],1))], axis=1), axis=1)
        return scs.norm.cdf(np.log(pred), scale=self.params[-1])
        

    def loge(self, Xe):
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.max(np.concatenate([np.dot(Xe, self.params[:-1])[:,np.newaxis], 1e-30*np.zeros((Xe.shape[0],1))], axis=1), axis=1)
        return np.max(np.concatenate([scs.norm.logcdf(np.log(pred), scale=self.params[-1])[:,np.newaxis], -1e99*np.ones((Xe.shape[0],1))], axis=1), axis=1)

    def log1e(self, Xe):
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.max(np.concatenate([np.dot(Xe, self.params[:-1])[:,np.newaxis], 1e-30*np.zeros((Xe.shape[0],1))], axis=1), axis=1)
        return scs.norm.logsf(np.log(pred), scale=self.params[-1])
    
    def fit(self, Xe, gamma, Y, w=1., warm_start=True, balance=False):
        if self.params is None:
            self.initialization(Xe, gamma, Y, w)
        if warm_start == False:
            start_params = None
        else:
            start_params = np.log(self.params.copy())
        if balance:
            q1, q2 = sum(Y)/sum(gamma), 1-sum(Y)/sum(gamma)
            c1, c2 = 1/q1/(1/q1 + 1/q2), 1/q2/(1/q1 + 1/q2)
            weights = c1*Y + c2*(1-Y)
        else:
            weights = 1
        self.model = LogProbitSig(Y, Xe, weights*gamma)
        self.res = self.model.fit(start_params=start_params, disp=0)
        self.params = np.exp(self.res.params.copy())

class GaussianSNPropensity(Propensity):
    def __init__(self):
        return super(GaussianSNPropensity, self).__init__()

    def __str__(self):
        if self.params is not None:
            s = 'Phi = {}'.format(self.params[:-1])
            s += '\n'
            s += 'sigma = {}'.format(self.params[-1])
            return s
        else:
            return ''

    def __repr__(self):
        return self.__str__()

    def initialization(self, Xe, gamma, Y, w=1.):
        self.params = np.random.rand(Xe.shape[1] + 1)
    
    def e(self, Xe):
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.max(np.concatenate([np.dot(Xe, self.params[:-1])[:,np.newaxis], 1e-30*np.zeros((Xe.shape[0],1))], axis=1), axis=1)
        return scs.norm.cdf(np.log(pred), scale=self.params[-1])
        

    def loge(self, Xe):
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.max(np.concatenate([np.dot(Xe, self.params[:-1])[:,np.newaxis], 1e-30*np.zeros((Xe.shape[0],1))], axis=1), axis=1)
        return np.max(np.concatenate([scs.norm.logcdf(np.log(pred), scale=self.params[-1])[:,np.newaxis], -1e99*np.ones((Xe.shape[0],1))], axis=1), axis=1)

    
    def logp(self, Xe):
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.max(np.concatenate([np.dot(Xe, self.params[:-1])[:,np.newaxis], 1e-30*np.zeros((Xe.shape[0],1))], axis=1), axis=1)
        return scs.norm.logpdf(np.log(pred), scale=self.params[-1])        

    def log1e(self, Xe):
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.max(np.concatenate([np.dot(Xe, self.params[:-1])[:,np.newaxis], 1e-30*np.zeros((Xe.shape[0],1))], axis=1), axis=1)
        return scs.norm.logsf(np.log(pred), scale=self.params[-1])
    
    def fit(self, Xe, gamma, Y, w=1., warm_start=True, balance=False):
        if self.params is None:
            self.initialization(Xe, gamma, Y, w)
        if warm_start == False:
            start_params = None
        else:
            start_params = np.log(self.params.copy())
        if balance:
            q1, q2 = sum(Y)/sum(gamma), 1-sum(Y)/sum(gamma)
            c1, c2 = 1/q1/(1/q1 + 1/q2), 1/q2/(1/q1 + 1/q2)
            weights = c1*Y + c2*(1-Y)
        else:
            weights = 1
        self.model = GaussianSN(Y, Xe, weights*gamma)
        self.res = self.model.fit(start_params=start_params, disp=0)
        self.params = np.exp(self.res.params.copy())

class LogisticPropensity(Propensity):
    '''
    Logistic Propensity : the feature vector Xe is assumed to contain an intercept as its first column
    '''
    def __init__(self):
        return super(LogisticPropensity, self).__init__()  

    def __str__(self):
        if self.params is not None:
            s = 'phi = {}'.format(self.params)
            return s
        else:
            return ''

    def __repr__(self):
        return self.__str__()

    def initialization(self, Xe, gamma, Y, w=1.):
        self.params = np.random.randn(Xe.shape[1]+1)
    
    def e(self, Xe):
        # add intercept
        Xe = np.concatenate([np.ones((Xe.shape[0],1)), Xe], axis=1)
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.dot(Xe, self.params)
        return scs.logistic.cdf(pred)
        

    def loge(self, Xe):
        # add intercept
        Xe = np.concatenate([np.ones((Xe.shape[0],1)), Xe], axis=1)
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.dot(Xe, self.params)
        return scs.logistic.logcdf(pred)

    def log1e(self, Xe):
        # add intercept
        Xe = np.concatenate([np.ones((Xe.shape[0],1)), Xe], axis=1)
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.dot(Xe, self.params)
        return scs.logistic.logsf(pred)
    
    def fit(self, Xe, gamma, Y, w=1., warm_start=True, balance=False):
        if self.params is None:
            self.initialization(Xe, gamma, Y, w)
        if warm_start == False:
            start_params = None
        else:
            start_params = self.params.copy()
        if balance:
            q1, q2 = sum(Y)/sum(gamma), 1-sum(Y)/sum(gamma)
            c1, c2 = 1/q1/(1/q1 + 1/q2), 1/q2/(1/q1 + 1/q2)
            weights = c1*Y + c2*(1-Y)
        else:
            weights = w
        # add intercept
        Xe = np.concatenate([np.ones((Xe.shape[0],1)), Xe], axis=1)
        self.model = sm.GLM(Y, Xe, freq_weights=weights*gamma, family=sm.families.Binomial())
        self.res = self.model.fit(start_params=start_params, disp=0)
        self.params = self.res.params.copy()

class GumbelPropensity(Propensity):
    def __init__(self):
        return super(GumbelPropensity, self).__init__()

    def __str__(self):
        if self.params is not None:
            s = 'Phi = {}'.format(self.params[:-1])
            s += '\n'
            s += 'k = {}'.format(self.params[-1])
            return s
        else:
            return ''

    def __repr__(self):
        return self.__str__()

    def initialization(self, Xe, gamma, Y, w=1.):
        self.params = np.random.rand(Xe.shape[1] + 1)   

    def e(self, Xe):
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.max(np.concatenate([np.dot(Xe, self.params[:-1])[:,np.newaxis], 1e-30*np.zeros((Xe.shape[0],1))], axis=1), axis=1)
        return scs.weibull_min.cdf(pred, self.params[-1])

    def loge(self, Xe):
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.max(np.concatenate([np.dot(Xe, self.params[:-1])[:,np.newaxis], 1e-30*np.zeros((Xe.shape[0],1))], axis=1), axis=1)
        return np.max(np.concatenate([scs.weibull_min.logcdf(pred, self.params[-1])[:,np.newaxis], -1e99*np.ones((Xe.shape[0],1))], axis=1), axis=1)

    def log1e(self, Xe):
        if self.params is None:
            print('Please initialise parameters first')
            return
        pred = np.max(np.concatenate([np.dot(Xe, self.params[:-1])[:,np.newaxis], 1e-30*np.zeros((Xe.shape[0],1))], axis=1), axis=1)
        return scs.weibull_min.logsf(pred, self.params[-1])
    
    def fit(self, Xe, gamma, Y, w=1., warm_start=True, balance=False):
        if self.params is None:
            self.initialization(Xe, gamma, Y, w)
        if warm_start == False:
            start_params = None
        else:
            start_params = np.log(self.params.copy())
        if balance:
            q1, q2 = sum(Y)/sum(gamma), 1-sum(Y)/sum(gamma)
            c1, c2 = 1/q1/(1/q1 + 1/q2), 1/q2/(1/q1 + 1/q2)
            weights = c1*Y + c2*(1-Y)
        else:
            weights = w
        self.model = Gumbel(Y, Xe, weights*gamma)
        self.res = self.model.fit(start_params=start_params, disp=0)
        # print(self.res.summary())
        self.params = np.exp(self.res.params.copy()) 


    
        