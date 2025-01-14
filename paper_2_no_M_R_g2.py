import numpy as np
import os 
import configparser
import matplotlib.pyplot as plt
import sys
import scipy
from scipy.integrate import quad
from scipy.interpolate import interp1d
import emcee
import corner
from scipy.stats import norm, gaussian_kde
from sympy import DiracDelta

# Set random seed for reproducibility
np.random.seed(42)


config = configparser.ConfigParser()
config.read('settings.ini')
main_dir=config['Main']['main_path']
plots_dir = os.path.join(main_dir,'plots')


def g(x):
    return (1/2+np.tan(np.pi*x/2))**2

def g2(x,beta=1.3):
    return beta**2/(beta+x)**2

def sample_data_from_function(n = 10,c=0.05,start=0,end=1,function=g):
    if function == g:
        x_values = np.linspace(0,1/np.pi,n+1,endpoint=True)[1:]
    elif function == g2:
        x_values = np.linspace(start,end,n,endpoint=True)
    #apply g(x) to get y values
    y_values = function(x_values)
    #apply the error to the values
    n = np.random.normal(loc=0,scale=1,size=n)
    d = y_values*(1+n*c)
    sigmas = c*d
    return x_values, d, sigmas


def sample_true_function(x_start=0,x_end=0.4,n=100,function=g):
    #generate x values
    x_values = np.linspace(x_start,x_end,n,endpoint=True)
    y_values = function(x_values)
    return x_values,y_values

def sample_estimated_function(a,x_values=None,x_start=0,x_end=0.4,n=100):
    if x_values is None:
        x_values = np.linspace(x_start,x_end,n,endpoint=True)
    else:
        pass
    y_values = [f_x_a(x,a) for x in x_values]
    y_values = np.array(y_values)
    return x_values,y_values


def estimate_prior(a,M,R):
    prefactor = (1/(np.sqrt(2*np.pi)*R))**(M+1)
    chi2_prior = np.sum(a**2)/R**2

    probability = prefactor*np.exp(-chi2_prior/2)
    return probability

def log_prior(a,M,R):
    chi2_prior = np.sum(a**2)/R**2
    log_prior = -((M+1)/2)*np.log(2*np.pi*R)-chi2_prior/2
    return log_prior


def f_x_a(x,a):
    f = 0
    for i in range(a.shape[0]):
        f += a[i]*x**i
    return f

def estimate_likelihood(x,d,sigmas,a):
    N = d.shape[0]
    prefactor = (1/np.sqrt(2*np.pi))**N*(1/np.prod(sigmas))
    chi2 = 0
    for i in range(N):
        chi2 += ((d[i]-f_x_a(x[i],a))/sigmas[i])**2
    probabiltiy = prefactor*np.exp(-chi2/2)
    if not np.isnan(probabiltiy):
        return probabiltiy
    else:
        return 0

def log_likelihood(a,x,d,sigmas):
    N = d.shape[0]
    chi2 = 0
    log_sigma_sum = 0
    for i in range(N):
        chi2 += ((d[i]-f_x_a(x[i],a))/sigmas[i])**2
        log_sigma_sum += np.log(1/sigmas[i])
    log_likelihood = N*np.log(1/np.sqrt(2*np.pi))+log_sigma_sum-(chi2/2)
    if not np.isnan(log_likelihood):
        return log_likelihood
    else:
        return -np.inf

def log_posterior(a,M,R,x,d,sigmas):
    return log_prior(a,M,R)+log_likelihood(a,x,d,sigmas)

def negative_log_posterior(a,M,R,x,d,sigmas):
    return -log_posterior(a,M,R,x,d,sigmas)

def simulate_posterior(M,R=1,samples=1000):
    x,d, sigmas = sample_data_from_function()
    posteriors = []
    a_s = []
    for i in range(samples):
        a = np.random.rand(M+1)*10
        a_s.append(a)
        prior = estimate_prior(a,M,R)
        likelihood = estimate_likelihood(x,d,sigmas,a)
        posteriors.append(prior*likelihood)
    return a_s,posteriors


def uniform_prior(a,lower_bound,upper_bound):
    n = len(a)
    # Check if all elements of the vector are within the bounds
    if np.all((a >= lower_bound) & (a <= upper_bound)):
        # Compute the joint probability
        prob = (1 / (upper_bound - lower_bound)) ** n
        return prob
    else:
        # If any element is outside the bounds, the probability is zero
        return -np.inf



class prior_a_class:
    def __init__(self,type):
        self.type = type

    def probability(self,a,a_bar):
        if self.type in ['A','B','A*']:
            a_bar = a_bar * np.ones(a.shape)
            pr_a_a_bar = np.heaviside(a_bar-np.abs(a),0)/(2*a_bar)
            pr_a_a_bar = np.prod(pr_a_a_bar)
            if not np.isnan(pr_a_a_bar):
                return pr_a_a_bar
            else:
                return 0
            
        elif self.type in ['C','C*']:
            factor = 1/(np.sqrt(2*np.pi)*a_bar)
            exponent = -(a**2)/(2*a_bar**2)
            prob_a = factor*np.exp(exponent)
            prob_a = np.prod(prob_a)
            if not np.isnan(prob_a):
                return prob_a
            else:
                return 0
            
    def log_probability(self,a,a_bar):
        if self.type in ['A','B','A*']:
            a_bar = a_bar * np.ones(a.shape)
            pr_a_a_bar = np.heaviside(a_bar-np.abs(a),0)/(2*a_bar)
            if np.all(pr_a_a_bar > 0):
                return np.sum(np.log(pr_a_a_bar))
            else:
                return -np.inf
        elif self.type in ['C','C*']:
            #could be exchanged by the full log expression later
            return np.log(self.probability(a,a_bar))
        
    def walker_initialization(self,a_bar,size):
        if self.type in ['A','B','A*']:
            return np.random.uniform(-a_bar,a_bar,size)
        elif self.type in ['C','C*']:
            return np.random.normal(0,a_bar,size)
        
    def generate_samples(self,a_bar_prior_object,n_samples=10000):
        a_bar_samples = a_bar_prior_object.generate_samples(n_samples=n_samples)

        a_samples = []
        for i in range(len(a_bar_samples)):
            if self.type in ['A','B','A*']:
                a_sample = np.random.uniform(-a_bar_samples[i],a_bar_samples[i],(1,1))
            elif self.type in ['C','C*']:
                a_sample = np.random.normal(0,a_bar_samples[i],(1,1))
            a_samples.append(a_sample)
        a_samples = np.vstack(a_samples)
        return a_samples
        

class prior_a_bar_class:
    def __init__(self,type,**kwargs):
        self.type = type
        for key,value in kwargs.items():
            setattr(self,key,value)

    def probability(self,a_bar):
        if self.type in ['A','C']:
            pr_a_bar = (np.heaviside(a_bar-self.a_smallereq,0)*np.heaviside(self.a_largereq-a_bar,0))/(np.log(self.a_largereq/self.a_smallereq)*a_bar)
        elif self.type in ['B']:
            e = np.exp(-(np.log(a_bar))**2/(2*self.sigma**2))
            pr_a_bar = e/(np.sqrt(2*np.pi)*a_bar*self.sigma)
        elif self.type in ['A*','C*']:
            pr_a_bar = DiracDelta(a_bar-self.a_fix)

        if not np.isnan(pr_a_bar):
            return pr_a_bar
        else:
            return 0
            
    def log_probability(self,a_bar):
        if self.type in ['A','C']:
            pr_a_bar = (np.heaviside(a_bar-self.a_smallereq,0)*np.heaviside(self.a_largereq-a_bar,0))/(np.log(self.a_largereq/self.a_smallereq)*a_bar)
            if not np.isnan(pr_a_bar):
                return np.log(pr_a_bar)
            else:
                return -np.inf

        elif self.type in ['B']:
            return np.log(self.probability(a_bar))
        elif self.type in ['A*','C*']:
            return np.log(self.probability(a_bar))
            
    def walker_initialization(self):
        if self.type in ['A','C']:
            return np.random.uniform(self.a_smallereq,self.a_largereq,1)
        elif self.type in ['B']:
            return np.random.lognormal(0,self.sigma,1)
        elif self.type in ['A*','C*']:
            return np.array([self.a_fix])
        
    def generate_samples(self,n_samples=10000,n_interpolation_accuracy=10000):
        if self.type in ['A','C']:
            vals = np.linspace(self.a_smallereq,self.a_largereq,n_interpolation_accuracy)
            cdf_vals = [quad(lambda x: self.probability(x), self.a_smallereq, val)[0] for val in vals]
            inverse_cdf = interp1d(cdf_vals,vals,bounds_error=False,fill_value='extrapolate')
            uniform_samples = np.random.uniform(0,1,n_samples)
            a_bar_samples = inverse_cdf(uniform_samples)
            return a_bar_samples
        elif self.type in ['B']:
            return np.random.lognormal(0,self.sigma,(n_samples,))
        elif self.type in ['A*','C*']:
            return np.ones((n_samples,))*self.a_fix



             

def log_posterior_marginalization_custom(params,x,y,sigmas,num_a_res,num_a_marg,prior_a,prior_a_bar,a_bar):
    if prior_a.type in ['C*','A*']:
        if num_a_marg >= 1:
            a_res = params[:num_a_res]
            a_marg = params[num_a_res:num_a_res+num_a_marg]

            

            log_prior = prior_a.log_probability(np.concatenate((a_res,a_marg)),a_bar)
            log_likelihood_a_res_marg = log_likelihood(np.concatenate((a_res,a_marg)),x,y,sigmas)

        else:
            a_res = params[:num_a_res]
            log_prior = prior_a.log_probability(a_res,a_bar)
            log_likelihood_a_res_marg = log_likelihood(a_res,x,y,sigmas)
        
        log_posterior_marg = log_prior +log_likelihood_a_res_marg

    elif prior_a.type in ['C','A']:
        if num_a_marg >= 1:
            a_res = params[:num_a_res]
            a_marg = params[num_a_res:num_a_res+num_a_marg]
            a_bar = params[-1]
            log_prior = prior_a.log_probability(np.concatenate((a_res,a_marg)),a_bar)
            log_likelihood_a_res_marg = log_likelihood(np.concatenate((a_res,a_marg)),x,y,sigmas)
            log_a_bar = prior_a_bar.log_probability(a_bar)

        else:
            a_res = params[:num_a_res]
            a_bar = params[-1]
            log_prior = prior_a.log_probability(a_res,a_bar)
            log_likelihood_a_res_marg = log_likelihood(a_res,x,y,sigmas)
            log_a_bar = prior_a_bar.log_probability(a_bar)
        
        log_posterior_marg = log_prior +log_likelihood_a_res_marg + log_a_bar


    if not np.isnan(log_posterior_marg):
        return log_posterior_marg
    else:
        return -np.inf




def marginalization(x_samples,y_samples,y_err,prior_a,prior_a_bar,M_max=9,r=3,num_samples=2000):
    all_a_res_samples = []
    num_a_res = r+1
    num_a_marg_total = M_Max-num_a_res+1

    
    if prior_a.type in ['C*','A*']:
        num_params = num_a_res+num_a_marg_total
        num_walkers = 10*num_params
        initial_pos = np.zeros((num_walkers,num_params))
        if num_a_marg_total >= 1:
            for i in range(num_walkers):
                #we should also adapt the initialization
                a_bar = prior_a_bar.a_fix
                initial_pos[i,:num_a_res] = prior_a.walker_initialization(a_bar,num_a_res)
                initial_pos[i,num_a_res:num_a_res+num_a_marg_total] = prior_a.walker_initialization(a_bar,num_a_marg_total)

        else:
            for i in range(num_walkers):
                #we should also adapt the initialization
                a_bar = prior_a_bar.a_fix
                initial_pos[i,:num_a_res] = prior_a.walker_initialization(a_bar,num_a_res)

    elif prior_a.type in ['C','A']:
        num_params = num_a_res+num_a_marg_total+1
        num_walkers = 10*num_params
        initial_pos = np.zeros((num_walkers,num_params))
        if num_a_marg_total >= 1:
            for i in range(num_walkers):
                #we should also adapt the initialization
                a_bar = prior_a_bar.walker_initialization()
                initial_pos[i,-1] = a_bar
                initial_pos[i,:num_a_res] = prior_a.walker_initialization(a_bar,num_a_res)
                initial_pos[i,num_a_res:num_a_res+num_a_marg_total] = prior_a.walker_initialization(a_bar,num_a_marg_total)

        else:
            for i in range(num_walkers):
                #we should also adapt the initialization
                a_bar = prior_a_bar.walker_initialization()
                initial_pos[i,-1] = a_bar
                initial_pos[i,:num_a_res] = prior_a.walker_initialization(a_bar,num_a_res)


            
    # Set up and run the sampler for the current M
    print('using custom prior')
    sampler = emcee.EnsembleSampler(num_walkers, num_params, log_posterior_marginalization_custom, args=(x_samples, y_samples, y_err,num_a_res,num_a_marg_total,prior_a,prior_a_bar,a_bar))
    sampler.run_mcmc(initial_pos, num_samples, progress=True)

    samples = sampler.get_chain(discard=200,flat=True)
    a_res_samples = samples[:, :num_a_res]
    all_a_res_samples.append(a_res_samples)


    all_a_res_samples = np.vstack(all_a_res_samples)


    # Calculate mean and standard deviation of each parameter in a_res
    means = np.mean(all_a_res_samples, axis=0)
    stds = np.std(all_a_res_samples, axis=0)

    print('means',np.round(means,2))
    print('stds',np.round(stds,2))

    return all_a_res_samples




#define parameters and priors
toy_function = g2
r = 3
M_Max = 7
prior_type = 'C*'
prior_defaults = {'a_smallereq':0.05,'a_largereq':20,'sigma':2,'a_fix':5}

prior_a = prior_a_class(type=prior_type)
prior_a_bar = prior_a_bar_class(type=prior_type,**prior_defaults)


x,y, sigmas = sample_data_from_function(start=0.1,end=0.1,function=toy_function,c=0.05)
x_true, y_true = sample_true_function(function=toy_function,x_start=0.01,x_end=0.25)


a_res_samples = marginalization(x,y,sigmas,prior_a,prior_a_bar,r=r,M_max=M_Max,num_samples=3000)



#histogram of posterior of a coeffifcient
coeff_order = 1
plt.hist(a_res_samples[:,coeff_order],bins=100,edgecolor='black',alpha=0.7,density=True)
plt.xlabel(f'Values a_{coeff_order}')
plt.ylabel('Frequency')
plt.title(f'Posterior of the coefficient a_{coeff_order}')
plt.show()
plt.savefig('hist_coefficient_posterior.png')
plt.clf()







# Create the corner plot
# Labels for the parameters
labels = [f'a_{i}'for i in range(a_res_samples.shape[1])]

fig = corner.corner(
    a_res_samples,
    labels=labels,
    show_titles=True,
    #range=[(0.2,0.35),(0,3),(-3.5,6),(-6,6),(-6,6)],
    #range=[(0.15,0.35),(0,3),(-8,10),(-18,18)],
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    quantiles=[0.16, 0.5, 0.84],  # Show the 16th, 50th, 84th percentiles
    plot_datapoints=True,
    fill_contours=True,
    levels=(0.68, 0.95),  # Confidence levels
    color="black",
    bins=100
)

# Show and save the plot
plt.show()
fig.savefig("corner_plot.png")
plt.clf()







####Plot the fit

plt.figure(figsize=(10, 6))

# True function
plt.plot(x_true, y_true, label="True Function", color="blue")

# Data points (noisy samples)
plt.errorbar(x, y, yerr=sigmas, fmt='o', color="cyan",ecolor='black', label="Sampled Data", capsize=3)

#Estimated model
a_res_mean = np.mean(a_res_samples, axis=0)
x_estim,y_estim = sample_estimated_function(a_res_mean,x_end=0.25)
plt.plot(x_estim,y_estim,label=f'Estimated Function, r = {r}, M_max = {M_Max}',color='red',linestyle='--')

#error bounds
random_indices = np.random.choice(a_res_samples.shape[0],size=1000,replace=False)
a_res_samples_subset = a_res_samples[random_indices]
sampling_x = np.linspace(0,0.25,100,endpoint=True)
processed_rows = []
for row in a_res_samples_subset:
    _ , sampling_y = sample_estimated_function(row,sampling_x)
    processed_rows.append(sampling_y)
model_predictive = np.stack(processed_rows)

model_quantiles = np.quantile(
    model_predictive, q=[0.025, 0.16, 0.84, 0.975], axis=0
)

# Fill the error bounds
plt.fill_between(sampling_x, model_quantiles[0], model_quantiles[-1], alpha=0.5, facecolor="C1",label="Model predictive distribution")
plt.fill_between(sampling_x, model_quantiles[1], model_quantiles[-2], alpha=0.5, facecolor="C1")

# Labels and legend
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Comparison of True Function, Sampled Data, and MCMC Model with Prior {prior_type}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('fitted_function.png')
plt.clf()

#Model fit test chi squared
def chi_squared(y,sigma_y,mu):
    return np.sum((y-mu)**2/sigma_y**2)

a_res_mean = np.mean(a_res_samples, axis=0)
_,mu = sample_estimated_function(a_res_mean,x)
chi_squared_value = chi_squared(y,sigmas,mu)
print('Chi-squared value:',np.round(chi_squared_value,2))
print('Chi-squared/dof:',np.round(chi_squared_value/(10-(r+1)),2))





#####make plot with prior,posterior and likelihood
coeff_order = 0

prior_samples = prior_a.generate_samples(prior_a_bar)
print(np.min(a_res_samples[:,coeff_order]),np.max(a_res_samples[:,coeff_order]))
all_likelihoods = []
for i in range(a_res_samples.shape[0]):
    a = a_res_mean.copy()
    a[coeff_order] = a_res_samples[i,coeff_order]
    all_likelihoods.append(estimate_likelihood(x,y,sigmas,a))
all_likelihoods = np.vstack(all_likelihoods)
all_likelihoods /= np.sum(all_likelihoods)


plt.figure(figsize=(12, 6))
#prior
plt.hist(prior_samples,bins=100,density=True,alpha=0.6,label='Prior')
#posterior
plt.hist(a_res_samples[:,coeff_order],bins=100,edgecolor='black',alpha=0.7,density=True,label='Posterior')
#likelihood
#plt.hist(all_likelihoods,bins=100,edgecolor='black',alpha=0.7,density=True,label='Likelihood')
plt.scatter(a_res_samples[:,coeff_order],all_likelihoods,label='Likelihood')
plt.title(f'a_{coeff_order}')
plt.legend()
plt.show()
plt.savefig('prior_posterior_likelihood.png')