import numpy as np
import matplotlib.pyplot as plt

def get_destination(pop_1):
    pop = pop_1.copy()
    counter = 1    
    for j in range(pop.shape[0]):        
        mn = np.where(min(pop)==pop)        
        pop[mn] = counter
        counter +=  1    
    pop -= 1
    return pop


def get_fitness(data, cost): # Функція оцінки
    
    population = data.copy()
    
    # Lower Bound
    if population.min()<0:
        population -=population.min()

    # Upper Bound   
    if population.max()>=1:
        population /= population.max()+np.random.rand()

    # Assign Destination
    for i in range(population.shape[0]):    
        population[i] = get_destination(population[i])

    # Compute Fitness    
    fitness = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        total_cost = 0
        for j in range(population.shape[1]):
            source, destination = j, int(population[i][j])             
            fitness[i] += cost[source][destination]       

    return fitness  

def single_point_crossover(population):
    r,c = population.shape[0], population.shape[1]            
    for i in range(0,r,2):        
        n = np.random.randint(1,c)        
        population[i], population[i+1] = np.append(population[i][0:n],population[i+1][n:c]),np.append(population[i+1][0:n],population[i][n:c])        
    return population 


def shift_mutation(population):
    if np.random.rand() < 0.03:
        r,c = population.shape[0], population.shape[1]            
        for i in range(r):
            n = np.random.randint(1,c)        
            population[i] = np.append(population[i][n:], population[i][:n] )
    return population      

# турнірний метод
def random_selection(population):
    r = population.shape[0]
    new_population = population.copy()    
    for i in range(r):        
        new_population[i] = population[np.random.randint(0,r)]
    return new_population


def ga_model(cost):
    n, c, max_iter = 100, cost.shape[0], 1000
    
    population = np.random.rand(n,c)
    fitness    = get_fitness(population, cost)        
    
    optimal_value    = fitness.min() 
    optimal_solution = population[np.where(fitness==optimal_value)][0]    
    arrayX = list()
    arrayY = list()
    arrayY.append(optimal_value)
    arrayX.append(0)
    counter = 0
    for i in range(max_iter):        
        counter = counter + 1
        arrayX.append(counter)
        population = random_selection(population)
        population = single_point_crossover(population)                        
        population = shift_mutation(population)        
        #print("Індивід:", population)
        fitness = get_fitness(population, cost)   
        arrayY.append(fitness.min())
        if fitness.min() < optimal_value:
            optimal_value    = fitness.min()
            optimal_solution = population[np.where(fitness==optimal_value)][0]                               
            
    plt.plot(arrayX,arrayY)
    return get_destination(optimal_solution), optimal_value


cost = np.array([[2,3,7,6,5,],
                 [3,2,5,9,8,],
                 [7,3,2,5,7,],
                 [5,3,2,8,7,],
                 [3,2,5,8,7]])

optimal_way, optimal_value = ga_model(cost)
print('Optimal Solution\n',optimal_way,
      '\nOptimal Value =' ,optimal_value)