'''
Implementations of the Zero-Order Methods for optimization. The algorithms are based on the Nelder-Mead simplex
and try to accurately estimate the true values of the parameters of the target environment. All the operations
are performed considering a barycenter including ALL the points, differently from the original algorithm that
considers only the best `n` points. 
'''

import numpy as np
import pickle as pkl
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Union
from pyDOE import lhs
from itertools import combinations
from .wrapper import ModelWrapper
from .utils import linear_map
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity

# Distribution Matching Adaptive Sampling
class Sampler(ABC):
    def __init__(self, 
                 args : dict, 
                 config_model : dict,
                 meta_wandb : dict,
                 meta_model : dict,
                 config_sampler : dict):
        
        # Maximum number of iterations for the algorithm
        self.iterations = 0
        # Configuration of the sampler
        self.config_sampler = config_sampler
        # Number of points in the simplex
        self.n_points = len(config_sampler['params']) + 1
        # Number of dimensions of the simplex
        self.n_dim = len(config_sampler['params']) 
        # Configuration of the algorithm
        self.args = args
        # Configuration of the model
        self.config_model = config_model
        # Meta-configuration of WandB
        self.meta_wandb = meta_wandb
        # Meta-configuration of the model
        self.meta_model = meta_model
        # History of vertices of the simplex
        self.solutions = np.zeros((args['max_iter'], self.n_points, self.n_dim))
        # History of fitnesses of the solutions of the simplex
        self.fitnesses = np.zeros((args['max_iter'], self.n_points))
        # Best solution found at each iteration
        self.best_solution = np.zeros((args['max_iter'], self.n_dim))
        # Best fitness found at each iteration
        self.best_fitness = np.zeros(args['max_iter']) - np.inf
        # Rewards of the solutions
        self.rewards = np.zeros((args['max_iter'], self.n_points), dtype=float)
        # Best rewards found at each iteration
        self.best_rewards = np.zeros(args['max_iter'], dtype=float) - np.inf
        # Domain of the sampling space
        self.count_operations = 0
        self.domain_min = np.array(args['domain_min'])
        self.domain_max = np.array(args['domain_max'])
    
    @abstractmethod
    def search(self) -> tuple:
        pass

    
class NelderMead(Sampler):
    def __init__(self, 
                 args : dict,
                 config_model : dict,
                 meta_wandb : dict,
                 meta_model : dict,
                 config_sampler : dict):
        '''
        Implementation of the Nelder-Mead algorithm for optimization. The algorithm is based on the simplex method
        and it is used to find a minimum (or maximum) of an objective function in a multidimensional space. Different
        tricks are used to avoid premature convergence and allow a better exploration until a good solution is found.

        Arguments
        ---------
        args : dict
            Dictionary containing the configuration parameters for the algorithm;
        config_model : dict
            Dictionary containing the configuration parameters for the model;
        meta_wandb : dict
            Dictionary containing the meta-configuration parameters for WandB;
        meta_model : dict
            Dictionary containing the meta-configuration parameters for the model;
        config_sampler : dict
            Dictionary containing the configuration parameters for the sampler.
        '''

        super().__init__(args, config_model, meta_wandb, meta_model, config_sampler)
        self._initialize(self.n_points, self.n_dim, self.args['initialization'])
        # Counter for the shrink operations
        self.count_shrink = 0

        if 'udr_lowhigh' in self.meta_model.keys():
            print(f"Vectorized envs with {self.meta_model['vectorized']} cores")
            print(f"UDR (percentages: {self.meta_model['udr_lowhigh']})")
            self.udr = self.meta_model['udr_lowhigh']

    def _update_worst_vertex(self,
                             new_vertex : np.ndarray, 
                             new_fitness : float,
                             new_avg : float) -> None:
        '''
        Update the worst vertex of the simplex replacing it with the new one, update its fitness and rank the
        vertices.

        Arguments
        ---------
        new_vertex : np.ndarray
            New vertex to replace the worst one;
        new_fitness : float
            Fitness of the new vertex;
        new_wrapper : ModelWrapper
            Wrapper of the new vertex;
        new_avg : float
            Average rewards of the new vertex.
        '''

        if new_fitness > self.simplex_fitness[self.vertices_rank[-1]]:
             print('New vertex is better than the best one')
        else:
             print('New vertex is not better than the best one')
        self.simplex[self.vertices_rank[0]] = deepcopy(new_vertex)              # Replace worst vertex with the new one
        self.simplex_fitness[self.vertices_rank[0]] = deepcopy(new_fitness)     # Update the fitness of the new vertex
        self.simplex_rewards[self.vertices_rank[0]] = deepcopy(new_avg)         # Update the rewards of the new verte
        self.vertices_rank = deepcopy(np.argsort(self.simplex_fitness))         # Rank the vertices based on their fitness

    def _sample_points(self,
                       n_points : int,
                       n_dim : int,
                       initialization : Union[str, np.ndarray]='lhs') -> None:
        '''
        Sample the initial points of the simplex. 

        Arguments
        ---------
        n_points : int
            Number of vertices in the simplex;
        n_dim : int
            Number of dimensions of the simplex;
        initialization : str | np.ndarray = 'lhs'
            Type of initialization. If a string is passed, it should be 'lhs'. In this case a Latin Hypercube
        '''
        
        good_init = False   # Flag to check if the initialization is good
        count_init = 0      # Counter for the number of initialization attempts

        print('Initializing simplex...')
        print('-----------------------')
        print('Number of points:', n_points)
        print('Number of dimensions:', n_dim)
        
        if type(initialization) == str:
            assert initialization == 'lhs', 'Invalid initialization type'
            
            print(f'Initialization type: {initialization}')
            print(f'Starting sampling domain: [{self.domain_min} - {self.domain_max}] x [{self.domain_min} - {self.domain_max}]')

            # Perform Latin Hypercube Sampling to initialize the simplex, but repeat the procedure if the points
            # are almost linearly dependent.
            while not good_init and count_init < self.args['max_init']:
                # Sample the points
                lhs_samples = lhs(n=n_dim, samples=n_points)
                # Scale the points to the domain
                lhs_samples = lhs_samples * (self.domain_max - self.domain_min) + self.domain_min
                # Compute all the combinations of the vertices
                comb = list(combinations(lhs_samples, 2))
                comb_vectors = list(combinations(comb, 2))
                # Compute the difference vectors for each pair
                difference_vectors = [np.array(pair[0][0]) - np.array(pair[0][1]) for pair in comb_vectors]
                # Compute the cosine similarity between each pair of difference vectors
                cosine_sim = cosine_similarity(difference_vectors)
                # Do not consider cosine similarity of a vector with itself
                cosine_sim[np.isclose(cosine_sim, 1)] = 0       
                
                # Compute ratio of aligned vectors to total unique pairs and repeat if the ratio is too high
                high_similarity_count = np.sum(np.abs(cosine_sim) > 0.90) / 2
                total_unique_pairs = (cosine_sim.size - cosine_sim.shape[0]) / 2
                good_init = (high_similarity_count / total_unique_pairs) < self.args['ratio_high_cosine']

                print(f'Ratio of aligned vectors: {high_similarity_count} / {total_unique_pairs}')
                print(f'Iteration {count_init + 1} - Good initialization: {good_init}')
                count_init += 1

            self.simplex, self.init_points = deepcopy(lhs_samples), deepcopy(lhs_samples)
        
        elif type(initialization) == np.ndarray:
            print('Initialization type: custom')
            assert initialization.shape == (n_points, n_dim), 'Invalid initialization shape'
            self.simplex = initialization
        
        else:
            raise ValueError('Invalid initialization type')
            
        self.simplex_fitness = np.zeros(n_points) - np.inf                  # Initialize fitness of each vertex
        self.simplex_rewards = np.zeros(n_points, dtype=float) - np.inf     # Initialize rewards of each vertex
        return lhs_samples
    
    def _starting_simplex_evaluation(self, lhs_samples) -> None:
        '''
        Evaluate all the vertexes of the initial simplex on the environment.

        Arguments
        ---------
        lhs_samples : np.ndarray
            Array containing the vertices of the simplex.
        '''
        
        print('\n\t--- STARTING SIMPLEX EVALUATION ---')
        
        # Train a model for each vertex of the simplex in order to avoid an initial bias when loading a pretrained one
        for (i, vertex) in enumerate(lhs_samples):
            wrapper_vertex, run = ModelWrapper.build_wrapper(
                self.config_model, self.meta_model, self.meta_wandb, self.config_sampler, vertex)
            
            vertex_avg_fitness, vertex_std_fitness, vertex_pessimistic = self._test(wrapper_vertex)                   # Test the vertex on the environment
            run.log({
                'avg_rew' : vertex_avg_fitness,
                'std_rew' : vertex_std_fitness,
                'pessimistic' : vertex_pessimistic,
                'operation' : float(-1)})
            run.finish()
            self.simplex_fitness[i] = deepcopy(vertex_pessimistic)              # Store the fitness of the vertex
            self.simplex_rewards[i] = deepcopy(vertex_avg_fitness)                        # Store the rewards of the vertex
            
            print(f'\tVertex {i} with fitness: {vertex_pessimistic}')
        
        self.vertices_rank = deepcopy(np.argsort(self.simplex_fitness))     # Rank the vertices based on their fitness
         
    def _initialize(self, 
                    n_points : int, 
                    n_dim : int, 
                    initialization : Union[str, np.ndarray]='lhs') -> None:
        '''
        Initialize starting simplex with `n_points` vertices in `n_dim` dimensions. The initialization
        can be done in different ways, such as `lhs` or by passing starting points as a matrix. 
        The vertices are tested on the environment and the fitness is ordered in ascending order.

        Arguments
        ---------
        n_points : int
            Number of vertices in the simplex;
        n_dim : int
            Number of dimensions of the simplex;
        initialization : str | np.ndarray = 'lhs'
            Type of initialization. If a string is passed, it should be 'lhs'. In this case a Latin Hypercube
            Sampling is performed. If a 'numpy' array is passed, it must have the shape '(n_points, n_dim)'.
        '''

        lhs_samples = self._sample_points(n_points, n_dim, initialization)
        self._starting_simplex_evaluation(lhs_samples)

    def _reflection(self, 
                    ro : float=1.0) -> None:
        '''
        Reflection step. Reflect the worst point of the simplex along the line connecting the barycenter
        and the worst point of a factor `ro`. If the new point is better than the best point, expand in that
        direction. If the new point is better than the second worst point but worse than the best point, replace
        the worst point with the reflection. If the new point is worse than the second worst point, contract the
        simplex.

        Arguments
        ---------
        ro : float = 1.0
            Reflection factor.
        '''

        print('\n\t--- REFLECTION ---')
        barycenter = self.simplex.mean(axis=0)                                  # Compute the barycenter of the simplex
        worst_point = self.simplex[self.vertices_rank[0]]                       # Get the worst point of the simplex
        best_point = self.simplex[self.vertices_rank[-1]]                       # Get the best point of the simplex
        best_fitness = self.simplex_fitness[self.vertices_rank[-1]]             # Get the fitness of the best vertex
        worst_fitness = self.simplex_fitness[self.vertices_rank[0]]             # Get the fitness of the worst vertex
        second_worst_fitness = self.simplex_fitness[self.vertices_rank[1]]      # Get the fitness of the second worst vertex   

        # Take worst point and reflect it
        reflection = barycenter + ro * (barycenter - worst_point)   
        reflection[reflection <= 0] = 0 + np.random.uniform(0.01, 0.06, reflection[reflection <= 0].shape)
        wrapper_reflection, run_reflection = ModelWrapper.build_wrapper(
            self.config_model, self.meta_model, self.meta_wandb, self.config_sampler, reflection)
        
        # Test the reflection on the environment
        reflection_avg_fitness, reflection_std_fitness, reflection_pessimistic = self._test(wrapper_reflection)
        run_reflection.log({
            'avg_rew' : reflection_avg_fitness,
            'std_rew' : reflection_std_fitness,
            'pessimistic' : reflection_pessimistic,
            'operation' : float(0)})
        run_reflection.finish()

        # If the reflection is better than the best point we are very happy and we expand in that direction
        if reflection_pessimistic > best_fitness:    
            self._expansion(reflection, barycenter, reflection_pessimistic, reflection_avg_fitness, 
                            wrapper_reflection, self.config_sampler['chi']) 

        # If reflection is better than second worst point but worse than best point we are still happy
        elif (reflection_pessimistic > second_worst_fitness) and (reflection_pessimistic <= best_fitness):
            self._update_worst_vertex(reflection, reflection_pessimistic, reflection_avg_fitness)

        # If reflection is worse than second worst point we need to contract the simplex
        elif reflection_pessimistic <= second_worst_fitness:
            self._contraction(reflection, barycenter, reflection_pessimistic, worst_fitness, worst_point,
                              best_point, self.config_sampler['gamma'])

    def _expansion(self, 
                   reflection : np.ndarray, 
                   barycenter : np.ndarray, 
                   reflection_fitness : float, 
                   reflection_avg_reward : float,
                   wrapper_reflection : ModelWrapper,
                   chi : float=2.0) -> None:
        '''
        Expansion step. Expand the reflection in the same direction as before. If the expansion is better than the
        reflection, replace the worst point with the expansion. Otherwise, replace the worst point with the reflection.

        Arguments
        ---------
        reflection : np.ndarray
            Reflected point;
        barycenter : np.ndarray
            Barycenter of the simplex;
        reflection_fitness : float
            Fitness of the reflected point;
        reflection_avg_reward : float
            Average rewards of the reflected point;
        wrapper_reflection : ModelWrapper
            Wrapper of the reflected point;
        chi : float = 2
            Expansion factor.
        '''
        
        print('\n\t--- EXPANSION ---')
        # Expand the reflection in the same direction as before
        expansion = barycenter + chi * (reflection - barycenter)
        expansion[expansion <= 0] = 0 + np.random.uniform(0.01, 0.06, expansion[expansion <= 0].shape)
        wrapper_expansion, run_expansion = ModelWrapper.build_wrapper(
            self.config_model, self.meta_model, self.meta_wandb, self.config_sampler, expansion)
        # Test the expansion on the environment
        expansion_avg_fitness, expansion_std_fitness, expansion_pessimistic = self._test(wrapper_expansion)
        run_expansion.log({
            'avg_rew' : expansion_avg_fitness,
            'std_rew' : expansion_std_fitness,
            'pessimistic' : expansion_pessimistic,
            'operation' : float(1)})
        run_expansion.finish()
        
        # If the expansion is better than the reflection we are very happy
        if expansion_pessimistic > reflection_fitness:
            self._update_worst_vertex(expansion, expansion_pessimistic,     # Replace worst point with expansion
                                      expansion_avg_fitness)   
        
        # If the expansion is not better than the reflection we are still happy but we put the reflection in the simplex
        else:
            self._update_worst_vertex(reflection, reflection_fitness, reflection_avg_reward)

    def _contraction(self,
                     reflection : np.ndarray,
                     barycenter : np.ndarray,
                     reflection_fitness : float,
                     worst_fitness : float,
                     worst_point : np.ndarray,
                     best_point : np.ndarray,
                     gamma : float=0.5) -> None:
        '''
        Contraction step. Contract the simplex towards the barycenter. If the contraction is better than the worst
        point, replace the worst point with the contraction. Otherwise, shrink the simplex towards the best point.

        Arguments
        ---------
        reflection : np.ndarray
            Reflected point;
        barycenter : np.ndarray
            Barycenter of the simplex;
        reflection_fitness : float
            Fitness of the reflected point;
        worst_fitness : float
            Fitness of the worst point;
        worst_point : np.ndarray
            Worst point of the simplex;
        best_point : np.ndarray
            Best point of the simplex;
        gamma : float = 0.5
            Contraction factor.
        '''

        print('\n\t--- CONTRACTION ---')
        # Compute the orientation point along which to contract
        orientation_point = worst_point if reflection_fitness < worst_fitness else reflection
        contraction = barycenter - gamma * (barycenter - orientation_point)
        contraction[contraction <= 0] = 0 + np.random.uniform(0.01, 0.06, contraction[contraction <= 0].shape)
        wrapper_contraction, run_contraction = ModelWrapper.build_wrapper(
            self.config_model, self.meta_model, self.meta_wandb, self.config_sampler, contraction)
        # Test the contraction on the environment
        contraction_avg_fitness, contraction_std_fitness, contraction_pessimistic = self._test(wrapper_contraction)
        run_contraction.log({
            'avg_rew' : contraction_avg_fitness,
            'std_rew' : contraction_std_fitness,
            'pessimistic' : contraction_pessimistic,
            'operation' : float(2)})
        run_contraction.finish()

        if contraction_pessimistic > worst_fitness:                             
            # Replace worst point with contraction
            self._update_worst_vertex(contraction, contraction_pessimistic, contraction_avg_fitness)
            # Reset the counter for the consecutive shrink operations
            self.count_shrink = 0
        else:
            # Shrink the simplex towards the best point
            self._shrink(best_point, self.config_sampler['sigma'])
            self.count_shrink += 1

    def _shrink(self, 
                best_point : np.ndarray, 
                sigma : float=0.5) -> None:
        '''
        Shrink step. Shrink all the vertices except the best one.

        Arguments
        ---------
        best_point : np.ndarray
            Best point of the simplex;
        sigma : float = 0.5
            Shrinking factor.
        '''

        # Add a pulse check to restart the simplex if the shrink is not working
        if self.args['pulse']:
            stop = self._pulse_check()
            if stop:
                return None

        print('\n\t--- SHRINK ---')
        # Shrink all the vertices except the best one towards the last
        for (i, idx) in enumerate(np.argsort(self.simplex_fitness)):
            if i == len(self.simplex_fitness) - 1:
                continue
            # Shrink the vertex towards the best point
            new_vertex = best_point + sigma * (self.simplex[idx] - best_point)
            new_vertex[new_vertex <= 0] = 0 + np.random.uniform(0.01, 0.06, new_vertex[new_vertex <= 0].shape)
            wrapper_new_vertex, run_new_vertex = ModelWrapper.build_wrapper(
                self.config_model, self.meta_model, self.meta_wandb, self.config_sampler, new_vertex)
            # Test the new vertex on the environment
            new_avg_fitness, new_std_fitness, new_pessimistic = self._test(wrapper_new_vertex)
            run_new_vertex.log({
                'avg_rew' : new_avg_fitness,
                'std_rew' : new_std_fitness,
                'pessimistic' : new_pessimistic,
                'operation' : float(3)})
            run_new_vertex.finish()
            
            self.simplex[idx] = deepcopy(new_vertex)                    # Replace the vertex with the new one
            self.simplex_fitness[idx] = deepcopy(new_pessimistic)       # Update the fitness of the vertex
            self.simplex_rewards[idx] = deepcopy(new_avg_fitness)       # Update the rewards of the vertex
        
        self.vertices_rank = deepcopy(np.argsort(self.simplex_fitness))

    def _pulse_check(self) -> bool:
        '''
        If we have shrinked too many times with no improvement, restart the simplex with a different sampling domain
        centered around the best point.

        Returns
        -------
        stop : bool
            Flag to stop the shrink operation.
        '''
    
        if self.count_shrink == self.args['max_shrink']:
            print('\n\n\t--- RESTARTING SIMPLEX ---')
            # Restart the simplex with a different sampling domain
            self.domain_min = self.simplex[self.vertices_rank[-1]] * (1 - self.args['expanding_domain_factor'])
            self.domain_max = self.simplex[self.vertices_rank[-1]] * (1 + self.args['expanding_domain_factor'])
            # self.meta_model['total_timesteps'] += 10_000
            self._initialize(self.n_points, self.n_dim, self.args['initialization'])
            self.count_shrink = 0
            return True
        return False

    def search(self) -> tuple:
        '''
        Search for the best approximate solution until the termination condition is met.

        Returns
        -------
        solutions : tuple
            Tuple containing statistics of the solutions found at each iteration;
        '''

        while not self._termination():
            
            print(f'\n--- ITERATION {self.iterations + 1} ---')
            
            vectors = self.simplex[1:] - self.simplex[0]    
            det = np.linalg.det(vectors)                    # Compute determinant of the simplex
            avg_fitness = np.mean(self.simplex_fitness)     # Compute average fitness of the simplex
            
            print(f'det = {det}')
            print(f'avg_fitness = {avg_fitness}')

            # If the simplex is becoming degenerate, add gaussian noise to the vertices. The noise is proportional
            # to the determinant and the average fitness of the simplex: if the fitness is low, the noise is high
            # as well, but as the points get closer to the solution the noise decreases, allowing for perturbations
            # only if the simplex is getting flat
            noises_std_det = np.array(
                [linear_map(np.abs(det), 
                            self.config_sampler['min_det'], 
                            self.config_sampler['max_det'] * 0.8, 
                            self.config_sampler['sigma_noise_min'][dim], 
                            self.config_sampler['sigma_noise_max'][dim], 
                            positive_slope=False) for dim in range(self.n_dim)]
                            )
            if np.abs(det) < self.config_sampler['max_det']:
                noise = np.random.normal(0, noises_std_det, self.simplex.shape)
                print(f'noise_det = {noises_std_det}')
                print(f'\nAdding noise to the simplex...')
                print(f'Noise: {noise}')
                # Add noise to the simplex except the best point
                for v_rank in range(self.n_points - 1):
                    self.simplex[self.vertices_rank[v_rank]] += noise[v_rank]
                
                self.simplex[self.simplex <= 0] = 0 + np.random.uniform(0.01, 0.06, self.simplex[self.simplex <= 0].shape)

            self._reflection(self.config_sampler['ro'])
            
            # Collect statistics
            self.solutions[self.iterations] = self.simplex
            self.fitnesses[self.iterations] = self.simplex_fitness
            self.best_solution[self.iterations] = self.simplex[self.vertices_rank[-1]]
            self.best_fitness[self.iterations] = self.simplex_fitness[self.vertices_rank[-1]]
            self.rewards[self.iterations] = self.simplex_rewards
            self.best_rewards[self.iterations] = self.simplex_rewards[self.vertices_rank[-1]]
            
            # If the best solution is improving, the shrinking is actually working
            if self.best_fitness[self.iterations] > self.best_fitness[self.iterations - 1]:
                self.count_shrink = 0

            self.iterations += 1

        with open(f'data/solutions.pkl', 'wb') as f:
                pkl.dump(self.solutions, f)
        with open(f'data/fitnesses.pkl', 'wb') as f:
                pkl.dump(self.fitnesses, f)
        with open(f'data/best_solution.pkl', 'wb') as f:
                pkl.dump(self.best_solution, f)
        with open(f'data/best_fitnesses.pkl', 'wb') as f:
                pkl.dump(self.best_fitness, f)
        with open(f'data/init_points.pkl', 'wb') as f:
                pkl.dump(self.init_points, f)
        with open(f'data/rewards.pkl', 'wb') as f:
                pkl.dump(self.rewards, f)
        with open(f'data/best_rewards.pkl', 'wb') as f:
                pkl.dump(self.best_rewards, f)
        with open(f'data/count_operations.pkl', 'wb') as f:
                pkl.dump(self.count_operations, f)
                

        print(f'\n\nBest solution found: {self.simplex[self.vertices_rank[-1]]}'
              f'\nFitness: {self.simplex_fitness[self.vertices_rank[-1]]}')
        
        return self.solutions, self.fitnesses, self.best_solution, self.init_points, self.count_operations

    def _termination(self) -> None:
        '''
        Termination conditions for the search. The search stops if the maximum number of iterations is reached
        or if the fitness of the best point is greater than the threshold.
        '''
        return (self.iterations == self.args['max_iter']) or (self.simplex_fitness[self.vertices_rank[-1]] >= self.args['threshold'])

    def _test(self, 
              vertex : ModelWrapper) -> None:
        '''
        Test on the environment to get the fitness of the vertex.

        Arguments
        ---------
        vertex : np.ndarray
            Vertex to test on the environment.
        '''

        avg_rew, std_rew, pessimistic = vertex.evaluate_policy()
        self.count_operations += 1
        return avg_rew, std_rew, pessimistic
    

class ApproximateNelderMead(NelderMead):
    def __init__(self, 
                 args : dict,
                 config_model : dict,
                 meta_wandb : dict,
                 meta_model : dict,
                 config_sampler : dict) -> None:
        super().__init__(args, config_model, meta_wandb, meta_model, config_sampler)
        self.approx_shrink_points = np.zeros((len(self.simplex) - 1, self.n_dim))
        self.approx_shrink_fitness = np.zeros(len(self.simplex_fitness) - 1)    
        self.inner_barycenters = np.zeros((len(self.simplex) - 1, self.n_dim))
        self.inner_barycenters_fitness = np.zeros(len(self.simplex_fitness) - 1)
        self.weight_by_distance = args['weight_by_distance']

    def _shrink(self, 
                best_point : np.ndarray, 
                sigma : float=0.5) -> None:
        '''
        Shrink step. Shrink all the vertices except the best one. Approximate the fitness of the shrinked vertex
        using a weighted average of the fitnesses of the vertex, the approximate fitness of the shrinked vertex 
        and the pivot.

        Arguments
        ---------
        best_point : np.ndarray
            Best point of the simplex;
        sigma : float = 0.5
            Shrinking factor.
        '''

        # Add a pulse check to restart the simplex if the shrink is not working
        if self.args['pulse']:
            stop = self._pulse_check()
            if stop:
                return None

        print('\n\t--- APPROXIMATE SHRINK ---')
        pivot = self.simplex.mean(axis=0)                                   # Compute the barycenter of the outer simplex
        wrapper_pivot, run_pivot = ModelWrapper.build_wrapper(
            self.config_model, self.meta_model, self.meta_wandb, self.config_sampler, pivot)
        # Test the pivot on the environment
        pivot_avg_fitness, pivot_std_fitness, pivot_pessimistic = self._test(wrapper_pivot)
        best_point_fitness = self.simplex_fitness[self.vertices_rank[-1]]   # Get the fitness of the best point
        weights = np.array([0.2, 0.2, 0.6])                                 # Weights for the average of the fitnesses

        for idx in np.argsort(self.simplex_fitness):
            # if i == len(self.simplex_fitness) - 1:
            if idx == len(self.simplex_fitness) - 1:
                continue

            shrinked_vertex = best_point + sigma * (self.simplex[idx] - best_point)     # Shrink the vertex towards the best point
            inner_symplex = np.array([self.simplex[idx],                                # Build the inner simplex
                                      shrinked_vertex, 
                                      pivot])
            
            # Use inner barycenter as an estimate of the new vertex
            self.simplex[idx] = inner_symplex.mean(axis=0)
            # Approximate the fitness of the shrinked vertex
            approx_shrinked_fitness = (1 - sigma) * self.simplex_fitness[idx] + sigma * best_point_fitness

            if not self.weight_by_distance:
                # Estimate the fitness of the inner barycenter using a weighted average of the fitnesses
                self.simplex_fitness[idx] = np.average([self.simplex_fitness[idx],                  
                                                        approx_shrinked_fitness,
                                                        pivot_pessimistic],
                                                        weights=weights)
            else:
                # Estimate the fitness of the inner barycenter using a KNN regressor
                model = KNeighborsRegressor(n_neighbors=inner_symplex.shape[0], weights='distance')
                model.fit(inner_symplex, [self.simplex_fitness[idx], 
                                          approx_shrinked_fitness, 
                                          pivot_pessimistic])
                self.simplex_fitness[idx] = model.predict([self.simplex[idx]])[0]
                self.simplex_rewards[idx] = model.predict([self.simplex[idx]])[0]

        self.vertices_rank = deepcopy(np.argsort(self.simplex_fitness))


def incomplete_reset(func):
    '''
    When reflecting the second worst point after having swapped them, we may find a better point than the worst one.
    In this case, we reset the counter for the incomplete shrink operations, since the shrink is actually working.
    '''
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        print("Found a point better than the worst\t(Reset incomplete shrink count)")
        self.count_incomplete_shrink = 0
    return wrapper


class IncompleteNelderMead(NelderMead):
    def __init__(self, 
                 args : dict,
                 config_model : dict,
                 meta_wandb : dict,
                 meta_model : dict,
                 config_sampler : dict) -> None:
        super().__init__(args, config_model, meta_wandb, meta_model, config_sampler)
        self.count_incomplete_shrink = 0

    _update_worst_vertex = incomplete_reset(NelderMead._update_worst_vertex)
    
    def _shrink(self,
                best_point : np.ndarray,
                sigma : float=0.5) -> None:
        '''
        Shrink step. Swap the worst and second worst points and add Gaussian noise to the worst point. If the
        number of shrink operations is a multiple of 3, test the worst point.

        Arguments
        ---------
        best_point : np.ndarray
            Best point of the simplex;
        sigma : float = 0.5
            Shrinking factor.
        '''
        
        if not (self.count_incomplete_shrink == 1):
            print('\n\t--- INCOMPLETE SHRINK (swap) ---')
            # Swap the worst and second worst points
            self.vertices_rank[0], self.vertices_rank[1] = self.vertices_rank[1], self.vertices_rank[0]
            self.simplex[self.simplex <= 0] = 0 + np.random.uniform(0.01, 0.06, self.simplex[self.simplex <= 0].shape)
        else:
            print('\n\t--- INCOMPLETE SHRINK (test) ---')
            self.vertices_rank = np.argsort(self.simplex_fitness)
            # Shrink only the worst point
            self.simplex[self.vertices_rank[0]] = best_point + sigma * (self.simplex[self.vertices_rank[0]] - best_point)
            wrapper_worst, run_worst = ModelWrapper.build_wrapper(
                self.config_model, self.meta_model, self.meta_wandb, self.config_sampler, self.simplex[self.vertices_rank[0]])
            new_avg_fitness, new_std_fitness, new_pessimistic = self._test(wrapper_worst)
            run_worst.log({
                'avg_rew' : new_avg_fitness,
                'std_rew' : new_std_fitness,
                'pessimistic' : new_pessimistic,
                'operation' : float(3)})
            run_worst.finish()
            self.simplex_fitness[self.vertices_rank[0]] = deepcopy(new_pessimistic)       # Update the fitness of the vertex
            self.simplex_rewards[self.vertices_rank[0]] = deepcopy(new_avg_fitness)       # Update the rewards of the vertex
            self.vertices_rank = deepcopy(np.argsort(self.simplex_fitness))
            self.count_incomplete_shrink = 0
            return None

        self.count_incomplete_shrink += 1
