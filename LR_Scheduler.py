import math

def step_decay(epoch):
   initial_lrate = 0.01
   drop = 0.1
   epochs_drop = 30.0
   lrate = initial_lrate * math.pow(drop, \
              math.floor(epoch/epochs_drop))
   return lrate
