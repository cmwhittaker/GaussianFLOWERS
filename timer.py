#%% A timer class to give nice feedback about the time remaining on long computations
import time
from humanfriendly import format_timespan

class stopwatch:
    def __init__(self):
        self.start_time = time.time()
        self.elapsed_time = 0
    def lap(self,frac,prefix_text): #like pressing lap on a stopwatch
        self.elapsed_time = time.time() - self.start_time
        self.remaining_time = (1/frac)*self.elapsed_time-self.elapsed_time #estimated_time - ellapsed time
        print(prefix_text + "{} elapsed, about {} left".format(format_timespan(self.elapsed_time),format_timespan(self.remaining_time)))
    def actual_time(self):
        #the actual time it took (above only estimates)
        return format_timespan(time.time()-self.start_time)