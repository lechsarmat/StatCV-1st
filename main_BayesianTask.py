import numpy as np
import func_BayesianTask as bt

p = 0.3
st_dict = bt.load_standards('data/standards')
or_key = str(np.random.choice(10, 1)[0])

bt.plot_task(or_key, st_dict, p, savepath = None)