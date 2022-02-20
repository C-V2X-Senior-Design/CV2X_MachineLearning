# Simulates Resource Blocks for ML Model
# Creates Dataset of N size to 'data' folder
# Splits data into two categories: test and train
# this uses code and logic from sim.py
import os
from random import randint, randrange, choice
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys
from tqdm import tqdm

N = 1000

class ResourcePoolSim:
    # Generates resource pool data for ML model
    SUBCHANNELS = 10
    SUBFRAMES = 10
    FRAMES = 5
    
    def __init__(self, _frameCount=FRAMES, _subChannelCount=SUBCHANNELS, _subFramesCount=SUBFRAMES):
        self.FRAMES = _frameCount
        self.SUBCHANNELS = _subChannelCount
        self.SUBFRAMES = _subFramesCount
        self.data = []
        print(f"Initialized Resouce Pool\n{self.FRAMES} Frames\t{self.SUBCHANNELS * self.SUBFRAMES} ({self.SUBCHANNELS} x {self.SUBFRAMES}) Resource Block(s)")
    
    def generateGrid(self, jamType=0):
        _frames = []*self.FRAMES
        jamSC = randrange(self.SUBCHANNELS)
        jamSF = randrange(self.SUBFRAMES)
        for i in range(self.FRAMES):
            # TODO random allocation of resource blocks, for now use 1-in use and 0-not used
            grid = [[1]*self.SUBFRAMES for _ in range(self.SUBCHANNELS)]

            if jamType == 1:
                # jam every frame with one "dead" pixel (1 -> 0)
                grid[jamSF][jamSC] = 0
                # jump by 1 pixel
                # TODO change from random to uniform
                jamSC = (jamSC + choice([1,0,-1]))%self.SUBCHANNELS
                jamSF = (jamSF + choice([1,0,-1]))%self.SUBFRAMES
            _frames.append(grid)
        self.data.append((_frames, bool(jamType)))
    
    def showGrid(self):
        # go through data to show resource pools and their frames
        for index, resource_pool in enumerate(self.data):
            print(f"\nResource Pool {index}\tJammed? {resource_pool[1]}")
            for f, _frame in enumerate(resource_pool[0]):
                print(f"Frame {f}")
                for subch in _frame:
                    print(subch)
    
    def clearGrid(self):
        # clears grid? Does this clear a specific resource pool or the whole data
        self.data = []
    
    def serializeGrid(self):
        # serialize data to a 1D array per resource pool for ML model
        # serialized_data is a 2D array with x resource pools and FRAMES*SUBCH*SUBF+1 for data. Last index is Jammed boolean
        self.serialized_data = np.zeros((len(self.data), self.FRAMES * self.SUBCHANNELS * self.SUBFRAMES + 1))
        for i in range(len(self.data)):
            j = 0
            for f, _frame in enumerate(self.data[i][0]):
                for subch in _frame:
                    for rb in subch:
                        self.serialized_data[i][j] = rb
                        j+=1
            self.serialized_data[i][j] = self.data[i][1] # label
    
    def writeGridToFile(self, serialized=False):
        # TODO rename for better convention
        DIR = "data/"
        if not os.path.isdir(DIR):
            os.mkdir(DIR)
        
        epoch_time = int(time.time())
        if serialized and self.serialized_data != None:
            self.serialized_data.tofile(f"{DIR}serialized_{epoch_time}.csv", sep=",")
        elif serialized and self.serialized_data == None:
            # serialize first
            self.serializeGrid()
            self.serialized_data.tofile(f"{DIR}serialized_{epoch_time}.csv", sep=",")
        else:
            df = pd.DataFrame(self.data)
            df.to_csv(f"{DIR}data_{epoch_time}.csv", sep=",")
    
    # TODO readGridFromFile()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 1000
    data = ResourcePoolSim()
    print(f"Running for {N} samples")
    for i in tqdm(range(N)):
        data.generateGrid(int(randint(0,1)))
        # data.serializeGrid()
    data.writeGridToFile()
    data.showGrid()
    data.clearGrid()