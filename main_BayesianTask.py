import numpy as np
import func_BayesianTask as bt

st_dict = bt.load_standards('data/standards')
or_key = str(np.random.choice(10, 1)[0])

print(f'List of all keys: {list(st_dict.keys())}')
print(f'Generated key: {or_key}')