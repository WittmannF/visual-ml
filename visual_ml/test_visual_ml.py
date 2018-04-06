import pandas as pd
import numpy as np
import visual_ml as vm

def test_create_X_grid():
	X = pd.DataFrame(np.ones([5,5]), columns=['A','B','C','D','E'])
	x = np.ones(5)
	X_map = vm.create_X_grid(X, x, ['D'])
	print("Input is {}, {} and ['B', 'D']".format(X, x))
	print("Output is {}".format(X_map))

def main():
	test_create_X_grid()

if __name__ == '__main__':
	main()