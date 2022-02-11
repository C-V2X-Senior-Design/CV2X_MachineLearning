#!/usr/bin/env python3
from random import randrange,choice

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
    jamSC = randrange(10) # subchanne l possition
    jamSF = randrange(10) # subframe
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
showGrids(test[1][0])

# TODO
# Def serialize():
# # turns input resource grid into single array

# Def clear grid()

# Def writeGridToFile()

# Def readGridFromFile()
# 