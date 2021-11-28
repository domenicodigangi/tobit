import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats
from scipy.stats import norm
from scipy.special import log_ndtr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error



def tobit_loss(x, y, coef, sigma, sample_weight=None, yl=0, yu=np.inf):

    const = 0.5 * np.log(2 * math.pi) + np.log(sigma)
    pred = np.dot(x, coef)
    pred = pred.ravel()
    diff = (y - pred) / sigma
    indl = (y == yl)
    indu = (y == yu)
    indmid = (y > yl) & (y < yu)
    if sample_weight is None:
        loss = (np.sum((diff[indmid] ** 2.0)/2 + const)
                - np.sum(norm.logcdf(diff[indl]))
                - np.sum(norm.logcdf(-diff[indu])))
    else:
        loss = (((np.sum(sample_weight[indmid]
                    * ((diff[indmid] ** 2.0) / 2 + const))
                    - np.sum(sample_weight[indl] * norm.logcdf(diff[indl]))
                    - np.sum(sample_weight[indu] * norm.logcdf(-diff[indu])))) 
                / sample_weight.sum())
    return loss

def tobit_loss_negative_gradient(x, y, coef, sigma, sample_weight=None, yl=0, yu=np.inf):

    pred = np.dot(x, coef)
    pred = pred.ravel()
    diff = (y - pred)/sigma
    indl = (y == yl)
    indu = (y == yu)
    indmid = (y > yl) & (y < yu)
    residual = np.zeros((y.shape[0],), dtype=np.float64)
    residual[indl] = (- np.exp(norm.logpdf(diff[indl])
                        - norm.logcdf(diff[indl])) / sigma)
    residual[indmid] = diff[indmid] / sigma
    residual[indu] = (np.exp(norm.logpdf(diff[indu])
                        - norm.logcdf(-diff[indu])) / sigma)

    if sample_weight is not None:
        residual = residual * sample_weight
    raise "missing grad wrt to sigma"
    return np.sum(residual)

def tobit_loss_hessian(self, y, pred, residual, **kargs):
    """Compute the second derivative """
    sigma = self.sigma
    sigma2 = self.sigma ** 2
    yl = self.yl
    yu = self.yu
    diff = (y - pred.ravel())/sigma
    indl = (y == yl)
    indu = (y == yu)
    indmid = (y > yl) & (y < yu)
    hessian = np.zeros((y.shape[0],), dtype=np.float64)
    lognpdfl = norm.logpdf(diff[indl])
    logncdfl = norm.logcdf(diff[indl])
    lognpdfu = norm.logpdf(diff[indu])
    logncdfu = norm.logcdf(-diff[indu])
    hessian[indmid] = 1/sigma2
    hessian[indl] = (np.exp(lognpdfl - logncdfl) / sigma2 * diff[indl]
                        + np.exp(2*lognpdfl-2 * logncdfl) / sigma2)
    hessian[indu] = (- np.exp(lognpdfu-logncdfu)/sigma2 * diff[indu]
                        + np.exp(2*lognpdfu-2 * logncdfu) / sigma2)
    raise "need to add sample weights"
    return hessian

def old_tobit_loss(xs, ys, params):
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b = params[:-1]
    # s = math.exp(params[-1])
    s = params[-1]

    to_cat = []

    cens = False
    if y_left is not None:
        cens = True
        left = (y_left - np.dot(x_left, b))
        to_cat.append(left)
    if y_right is not None:
        cens = True
        right = (np.dot(x_right, b) - y_right)
        to_cat.append(right)
    if cens:
        concat_stats = np.concatenate(to_cat, axis=0) / s
        log_cum_norm = scipy.stats.norm.logcdf(concat_stats)  # log_ndtr(concat_stats)
        cens_sum = log_cum_norm.sum()
    else:
        cens_sum = 0

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        mid = scipy.stats.norm.logpdf(mid_stats) - math.log(max(np.finfo('float').resolution, s))
        mid_sum = mid.sum()
    else:
        mid_sum = 0

    loglik = cens_sum + mid_sum

    return - loglik


def old_tobit_loss_der(xs, ys, params):
    x_left, x_mid, x_right = xs
    y_left, y_mid, y_right = ys

    b = params[:-1]
    # s = math.exp(params[-1]) # in censReg, not using chain rule as below; they optimize in terms of log(s)
    s = params[-1]

    beta_jac = np.zeros(len(b))
    sigma_jac = 0

    if y_left is not None:
        left_stats = (y_left - np.dot(x_left, b)) / s
        l_pdf = scipy.stats.norm.logpdf(left_stats)
        l_cdf = log_ndtr(left_stats)
        left_frac = np.exp(l_pdf - l_cdf)
        beta_left = np.dot(left_frac, x_left / s)
        beta_jac -= beta_left

        left_sigma = np.dot(left_frac, left_stats)
        sigma_jac -= left_sigma

    if y_right is not None:
        right_stats = (np.dot(x_right, b) - y_right) / s
        r_pdf = scipy.stats.norm.logpdf(right_stats)
        r_cdf = log_ndtr(right_stats)
        right_frac = np.exp(r_pdf - r_cdf)
        beta_right = np.dot(right_frac, x_right / s)
        beta_jac += beta_right

        right_sigma = np.dot(right_frac, right_stats)
        sigma_jac -= right_sigma

    if y_mid is not None:
        mid_stats = (y_mid - np.dot(x_mid, b)) / s
        beta_mid = np.dot(mid_stats, x_mid / s)
        beta_jac += beta_mid

        mid_sigma = (np.square(mid_stats) - 1).sum()
        sigma_jac += mid_sigma

    combo_jac = np.append(beta_jac, sigma_jac / s)  # by chain rule, since the expression above is dloglik/dlogsigma

    return -combo_jac


def k_fun(t, t0, H, type="epa"):
    if type == "epa":
        k = 1 - ((t-t0)**2)/(H**2) 
    if type == "exp":
        k = np.exp((t-t0)/H) 
    if type == "gauss":
        k = np.exp(-(t-t0)**2/H**2) 

    k[k<0] = 0
    return k
class TobitModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.ols_coef_ = None
        self.ols_intercept = None
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None

    def fit(self, x, y, type=None, bandwidth = 20, verbose=False):
        """
        Fit a maximum-likelihood Tobit regression
        :param x: Pandas DataFrame (n_samples, n_features): Data
        :param y: Pandas Series (n_samples,): Target
        :param verbose: boolean, show info from minimization
        :return:
        """

        if type is None:
            sample_weight=None
        else:
            T_train = y.shape[0]
            sample_weight = k_fun(np.array(range(T_train)), bandwidth, 20, type=type)
        x_copy = x.copy()
        if self.fit_intercept:
            x_copy.insert(0, 'intercept', 1.0)
 
        init_reg = LinearRegression(fit_intercept=False).fit(x_copy, y)
        b0 = init_reg.coef_
        self.b0 = b0
        y_pred = init_reg.predict(x_copy)
        resid = y - y_pred
        resid_var = np.var(resid)
        s0 = np.sqrt(resid_var)
        params0 = np.append(b0, s0)
        self.params0 = params0
        # print(tobit_loss(x_copy, y, params0[:-1], params0[-1]))
        result = minimize(lambda params: tobit_loss(x_copy, y, params[:-1], params[-1], sample_weight=sample_weight), params0, method='BFGS', options={'disp': verbose})
        if verbose:
            print(result)
        self.ols_coef_ = b0[1:]
        self.ols_intercept = b0[0]
        if self.fit_intercept:
            self.intercept_ = result.x[1]
            self.coef_ = result.x[1:-1]
        else:
            self.coef_ = result.x[:-1]
            self.intercept_ = 0
        self.sigma_ = result.x[-1]
        return self

    def predict(self, x):
        return self.intercept_ + np.dot(x, self.coef_)

    def score(self, x, y, scoring_function=mean_absolute_error):
        y_pred = np.dot(x, self.coef_)
        return scoring_function(y, y_pred)
