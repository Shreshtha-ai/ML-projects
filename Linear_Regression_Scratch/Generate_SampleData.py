import math
import random

random.seed(42) # it gives same starting point every time 

TRUE_W = 3.0 # HERE W IS WEGHT - SLOPE 
TRUE_B = 1.0 # HERE B IS BIAS - INTERCEPT - Y 

N_SAMPLES = 100 #GENERATE 100 DATA POINTS 

X = [random.uniform(0, 10) for i in range(N_SAMPLES)] #Generating 100 random values between 0 and 10 
Y = [TRUE_W*x + TRUE_B+ random.gauss(0,2.0) for x in X] #Generating 100 random values between 0 and 10 (adding some noise)
print(f"Generated {N_SAMPLES} samples")

print(f" True relationship: y ={TRUE_W}x +{TRUE_B}(+noise)")

print(f"First 5 points: {[(round(X[i], 2), round(Y[i], 2)) for i in range(5)]}")

