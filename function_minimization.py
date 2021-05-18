from numpy.random import randint
from numpy.random import rand
import numpy
import math
import matplotlib.pyplot as plt

count_of_pop = 100
n_iter = 100
n_bits = 16
r_cross = 0.8
r_mut = 0.1 

bound = [[-2.0, 2.0], [0.0, 1.0]]


def funcionX1X2(x):
    return math.sin(10 * x[0]) + x[0] * math.cos(2 * numpy.pi * x[1])

# Турнірний метод
def selection(pop, scores, k=3):
    # беремо рандомне число 
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # порівнюємо функції оцінки кожного індивіда з вибраним рандомним
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# Схрещення 
def crossover(p1, p2, r_cross):
    # дефолтне копіювання двох батьків
    c1, c2 = p1.copy(), p2.copy()
    # Вірогідність схрещення
    if rand() < r_cross:
        pt = randint(1, len(p1)-2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
        
    return [c1, c2]


# Мутація
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # ймовірність мутації
        if rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]

def decode(bound, n_bits, bitstring):
    decoded = []
    largest = 2 ** n_bits
    for i in range(len(bound)):
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(b) for b in substring])
        int_value = int(chars, 2)
        nrm_value = bound[i][0] + (int_value/largest) * (bound[i][1] - bound[i][0])
        decoded.append(nrm_value)
    
    return decoded


def StartGeneticAlgotithm(funcionX1X2, bound=bound, decode_func=decode, n_bits=n_bits, n_iter=n_iter, count_of_pop=count_of_pop, r_cross=r_cross, r_mut=r_mut):
    pop = [randint(0, 2, n_bits*len(bound)).tolist() for _ in range(count_of_pop)]
    decoded_pop = [decode_func(bound, n_bits, p) for p in pop]

    best, best_eval = 0, funcionX1X2(decoded_pop[0])
    xpoints = list()
    ypoints = list()
    counter = 0
    for gen in range(n_iter):
        counter +=1
        # розрахунок кандидатів в конкретній популяції
        scores = [funcionX1X2(c) for c in decoded_pop]
        # Знаходження кращого розв'язку
        for i in range(count_of_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                xpoints.append(counter)
                ypoints.append(scores[i])
        #print(f"Номер популяції: {gen}, [x1,x2]: ({bound,n_bits,pop[i]}), Функція оцінки: {scores[i]:.03f}")
        print(f"Номер популяції: {gen}, [x1,x2]: ({decode(bound,n_bits,pop[i])}), Функція оцінки: {scores[i]:.03f}")

        selected = [selection(pop, scores) for _ in range(count_of_pop)]
        new_pop = []
        for i in range(0, count_of_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                new_pop.append(c)
                    
        pop = new_pop
        decoded_pop = [decode_func(bound, n_bits, p) for p in pop]
    plt.plot(xpoints,ypoints)
    return [best, best_eval]


best, score = StartGeneticAlgotithm(funcionX1X2)
print("Best individ: ")
decoded = decode(bound, n_bits, best)
print(f"({decoded}) = {score:.07f}")