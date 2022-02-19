#!/usr/bin/env python3
from random import randrange,choice
import matplotlib.pyplot as plt
import numpy as np

print("hello cv2x")

SUBCHANNELS = 10
SUBFRAMES   = 10
FRAMES = 5

def showGrids(res_pool):
# prints grid usage for visual inspection
    for f,_frame in enumerate(res_pool):
        print("frame {}".format(f))
        for subch in _frame:
            print(subch)
            # for rb in subch:
            #     print(subch,end='')
            # print('')

def generate_grid(frames, jamType=0):
    _frames = []*frames
    jamSC = randrange(SUBCHANNELS) # subchanne l possition
    jamSF = randrange(SUBFRAMES) # subframe
    for i in range(frames):
        # TODO random allocation
        # for now assume grid in is full use
        # use integers: 1-in use or 0-not used
        # instead of booleans
        grid = [[1]*SUBFRAMES for _ in range(SUBCHANNELS)]

        if jamType==1: # narrow band jammer
            # jam every frame with one pixel,
            # ASSUME once jammed RB stays zero for a long time after
            # TODO we should think how to model this...
            grid[jamSF][jamSC] = 0

            #  jump by 1 pixel
            # TODO wrap in a smarter way, more like what happens in reality
            # TODO change ranom walk to explore whole grid more uniformly
            jamSC = (jamSC + choice([1,0,-1]))%10
            jamSF = (jamSF + choice([1,0,-1]))%10
        _frames.append(grid)
    return _frames
# label test data: (resource_pool, isJammed)
# make sute to shuffle jammed/normal cases before training
test = []
# no jam test cases
run = (generate_grid(FRAMES,jamType=0), False)
test.append(run)
# jammed test cases
run = (generate_grid(FRAMES,jamType=1), True)
test.append(run)
showGrids(test[0][0])
showGrids(test[1][0])

# TODO
# Def serialize():
# turns input resource grid into single array
def serialize(res_pool):
    print(res_pool)
    # print(len(res_pool))
    # print(len(res_pool[0]))
    serialized_resource = np.zeros((len(res_pool), len(res_pool[0])**2))
    # print(serialized_resource)

    for f, _frame in enumerate(res_pool):
        print(f"Serializing frame {f}")
        i = 0
        for subch in _frame:
            for rb in subch:
                serialized_resource[f][i] = rb
                i+=1
    # print(serialized_resource) # 2D array, needs to be flatten to become 1D.
    return serialized_resource

serialize(test[0][0])
serialize(test[1][0])

print("RESOURCE POOL\n")
print(test)

# Def clear grid()

# Def writeGridToFile()

# Def readGridFromFile()
# 