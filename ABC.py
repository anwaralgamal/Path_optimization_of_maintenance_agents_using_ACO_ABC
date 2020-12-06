

from ecabc.abc import *
import os
import time

def rastrigin(values3, args=None):
   x=values3[0]
   d=15
   for m in range(1,d,1):
    sum = np.sum([20 + np.power(x, 2) - 10 * (np.cos(2 * np.pi * x))])
    
   cost=(10*d)+sum
   if args:
        print(args)

   time.sleep(1)
   print("cost="+str(cost))
   return cost

def ackley(values3, args=None):
    a=20
    b=0.2
    c=2*np.pi
    x=values3[0]
    sum_part1 = x**2 
    part1 = -1.0 * a * np.exp(-1.0 * b * np.sqrt((0.5 * sum_part1)))

    sum_part2 = np.cos(c * x)
    part2 = -1.0 * np.exp((0.5 * sum_part2))
    cost=a + np.exp(1) + part1 + part2
    print("cost="+str(cost))
    return cost


def sphere(values3, args=None):
    
    d=15
    x=values3[0]
    
    cost=x**2
    if args:
        print(args)

    time.sleep(1)
    print("cost="+str(cost))
    return cost
def beale(values4, args=None):
    x=values4[0]
    y=values4[1]
    part1=1.5-x+(x*y)
    part2=2.25-x+(x*(y**2))
    part3=2.625-x+(x*(y**3))
    cost=abs((part1**2)+(part2**2)+(part3**3))
    if args:
        print(args)

    time.sleep(1)
    print("cost="+str(cost))
    return cost

def levi_13(values5, args=None):
    x=values5[0]
    y=values5[1]
    part1= (np.sin(3*x*np.pi))**2
    part2=(x-1)**2
    part3=1+(np.sin(3*y*np.pi))**2
    part4=(y-1)**2
    part5=1+(np.sin(2*y*np.pi))**2
    cost=abs(part1+(part2*part3)+(part4*part5))
    if args:
        print(args)

    time.sleep(1)
    print("cost="+str(cost))
    return cost
def data(values6, args=None):
    
    x=values6[0]
    cost=1.0832989432232585*x+0.03376599150401478
    if args:
        print(args)

    time.sleep(1)
    print("cost="+str(cost))
    return cost
     
    


if __name__ == '__main__':
                # First value      # Second Value     # Third Value      # Fourth Value
    values = [('int', (0,100)), ('int', (0,100)), ('float',(0,100)), ('float', (0, 100))]
    values2 = [('int', (20,50))]
    values3 = [('int', (-50,50))]
    values4 = [('float', (-4.5,4.5)),('float', (-4.5,4.5))]
    values5 = [('int', (-10,10)),('int', (-10,10))]
    values6 = [('float', (32.02052, 32.14307))]
    
    
    start = time.time()
    abc = ABC(fitness_fxn=data, 
            value_ranges=values6
            )
    abc.create_employers()
    while True:
        abc.save_settings('{}/settings.json'.format(os.getcwd()))
        abc._employer_phase()
        abc._calc_probability()
        abc._onlooker_phase()
        abc._check_positions()
        abc._cycle_number = abc._cycle_number + 1
        if (abc.best_performer[0] < 2):
            break
        if (abc._cycle_number == 10):
            break
    data(abc.best_performer[1], abc.best_performer[1])
    print("fitness="+str(abc.best_performer[0]))
    print("minumum x, y="+str(abc.best_performer[1][0]))
    print("execution time = {}".format(time.time() - start))
    

