from numpy.random import multivariate_normal
from cosserat_rods import*
class CMAES:
    """Naive CMA implementation"""
    
    def __init__(self, initial_mean, sigma, popsize, **kwargs):

        self.centroid = np.asarray(initial_mean).copy()
        self.sigma = sigma
        self.pc = np.zeros_like(initial_mean)
        self.ps = np.zeros_like(initial_mean)        
        self.C = np.eye(initial_mean.shape[0])
        self.B = np.eye(self.C.shape[0])
        self.diagD = np.ones(initial_mean.shape[0])
        
        # Optimal popsize 
        self.popsize = popsize
        self.mu = popsize // 2
        
        # Update weights later on
        # Constant weight policy
        # self.weights = np.ones((self.mu, )) / self.mu

        # Decreasing weight policy
        self.weights = np.arange(self.mu, 0.0, -1.0)
        self.weights /= np.sum(self.weights)
        
        # Negative, Positive weight policy
        # unscaled_weights = np.arange(1.0 ,  1.0 + popsize)
        # unscaled_weights = np.log(0.5 * (popsize + 1.0) / unscaled_weights)

        # Utility variables
        self.dim = initial_mean.shape[0]

        # Expectation of a normal distribution
        self.chiN = np.sqrt(self.dim) * (1.0 - 0.25 / self.dim + 1.0/(21.0 * self.dim**2))
        self.mueff = 1.0 / np.linalg.norm(self.weights, 2)**2
        self.generations = 0
        
        # Options
 
        # Sigma adaptation
        # cs is short for c_sigma
        self.cs = kwargs.get("cs", (2.0 + self.mueff) / (self.dim + self.mueff + 5.0))
        # ds is short for d_sigma
        self.ds = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0)/ (self.dim + 1.0)) - 1.0) + self.cs
        
        # Covariance adaptation
        self.cc = kwargs.get("cc", (4.0 + self.mueff/self.dim) / (self.dim + 4.0 + 2.0 * self.mueff/self.dim))
        self.ccov = 0.0
        # If implementing the latest version of CMA according to the tutorial, 
        # these parameters can be useful
        self.ccov1 = 2.0 / ((self.dim + 1.3)**2 + self.mueff)
        self.ccovmu = min(1.0 - self.ccov1, 2.0 * (self.mueff - 2.0 + 1.0/self.mueff)/((self.dim + 2.0)**2 + self.mueff))
        
        
        self.stats_centroids = []
        self.stats_new_centroids = []
        self.stats_covs = []
        self.stats_new_covs = []
        self.stats_offspring = []
        self.stats_offspring_weights = []
        self.stats_ps = []
    
    def update(self, problem, population):
       
        # -- store current state of the algorithm
        self.stats_centroids.append(copy.deepcopy(self.centroid))
        self.stats_covs.append(copy.deepcopy(self.C))

        pop_fitness = np.zeros(self.popsize)
        for idx in range(pop_size):
            pop_fitness[idx] = problem.calc_fitness(b_coeff_and_lambda=population[idx])
        index = np.argsort(pop_fitness)
        population = population[index]
        # population.sort(key=lambda ind: problem(ind[0], ind[1]))
        # population.sort(key=problem)
        
        # -- store sorted offspring
        self.stats_offspring.append(copy.deepcopy(population))
        
        old_centroid = self.centroid
        # Note : the following does m <- <x>_w
        # Note : this is equivalent to doing m <- m + sigma * <z>_w
        # as x = m + sigma * z provided the weights sum to 1.0 which it
        # does
        self.centroid = np.dot(self.weights, population[0:self.mu])
        
        # -- store new centroid
        self.stats_new_centroids.append(copy.deepcopy(self.centroid))
        
        c_diff = self.centroid - old_centroid
        
        # Cumulation : update evolution path
        # Equivalent to in-class definition
        self.ps = (1 - self.cs) * self.ps \
             + np.sqrt(self.cs * (2 - self.cs) * self.mueff) / self.sigma \
             * np.dot(self.B, (1. / self.diagD) * np.dot(self.B.T, c_diff))
        
        # -- store new evol path
        self.stats_ps.append(copy.deepcopy(self.ps))
        
        hsig = float((np.linalg.norm(self.ps) / 
                np.sqrt(1. - (1. - self.cs)**(2. * (self.generations + 1.))) / self.chiN
                < (1.4 + 2. / (self.dim + 1.))))
        
        self.pc = (1 - self.cc) * self.pc + hsig \
                  * np.sqrt(self.cc * (2 - self.cc) * self.mueff) / self.sigma \
                  * c_diff
        
        # Update covariance matrix
        artmp = population[0:self.mu] - old_centroid
        self.C = (1 - self.ccov1 - self.ccovmu + (1 - hsig) \
                   * self.ccov1 * self.cc * (2 - self.cc)) * self.C \
                + self.ccov1 * np.outer(self.pc, self.pc) \
                + self.ccovmu * np.dot((self.weights * artmp.T), artmp) \
                / self.sigma**2
        
        # -- store new covs
        self.stats_new_covs.append(copy.deepcopy(self.C))
        
        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1.) \
                                * self.cs / self.ds)
        
        self.diagD, self.B = np.linalg.eigh(self.C)
        indx = np.argsort(self.diagD)
        
        self.cond = self.diagD[indx[-1]]/self.diagD[indx[0]]
        
        self.diagD = self.diagD[indx]**0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD
        
    def run(self, problem):
        # At the start, clear all stored cache and start a new campaign
        self.reset()
        
        for i in tqdm(range(40)):
            # Sample the population here!
            population = list(multivariate_normal(self.centroid, self.sigma**2 * self.C, self.popsize))
            populations = np.array(population)
            # Pass the population to update, which computes all new parameters 
            self.update(problem, populations)
            # print(np.array(population).shape)
            self.generations += 1
            print(f"generation {self.generations}")
        else:
            return population[0]
        
    def reset(self):
        # Clears everything to rerun the problem
        self.stats_centroids = []
        self.stats_new_centroids = []
        self.stats_covs = []
        self.stats_new_covs = []
        self.stats_offspring = []
        self.stats_offspring_weights = []
        self.stats_ps = []

class SnakeProblem:
    def __init__(self):
        pass
    def calc_fitness(self,b_coeff_and_lambda):
        b_coeff = b_coeff_and_lambda[:-1]
        lambda_m = b_coeff_and_lambda[-1]
        distance_traveled = run_snake(b_coeff=b_coeff,wave_length=lambda_m,n_elements=10,run_time=2)
        boundary = 50
        b_coeff = np.array([b_coeff])
        fitness = distance_traveled
        abs_coeff = np.abs(b_coeff)
        penalty = 1000
        if np.any(abs_coeff) > boundary:
            for idx in np.where(np.abs(b_coeff) > 50):
                fitness -= np.abs(b_coeff)[idx] * penalty
        return fitness

initial_mean = np.array([0,25,25,25,25,0.97])
sigma = 1
pop_size = 6
snake_optimization = CMAES(initial_mean=initial_mean,sigma=sigma,popsize=pop_size)
snake_problem = SnakeProblem()
answer = snake_optimization.run(snake_problem)
print(answer)