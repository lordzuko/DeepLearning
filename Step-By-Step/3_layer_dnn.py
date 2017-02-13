import numpy as np

np.random.seed(1)

def relu(x):
	return (x > 0)*x

def relu2deriv(output):
	return output > 0

alpha = 0.2
hidden_size = 4

streetlights = np.array([[1,0,1],
						[0,1,1],
						[0,0,1],
						[1,1,1]
			])

walk_vs_stop = np.array([[1,1,0,0]]).T

weights_0_1 = 2*np.random.random((3,hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size,1)) - 1

for iteration in range(100):
	layer_2_error = 0
	for i in range(len(streetlights)):
		layer_0 = streetlights[i:i+1] # 1x3
		layer_1 = relu(layer_0.dot(weights_0_1)) # (1x3) x (3x4) ->1x4
		layer_2 = layer_1.dot(weights_1_2)	# (1x4)x(4x1) -> 1x1

		layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)

		layer_2_delta = (layer_2 - walk_vs_stop[i:i+1]) # 1x1
		# layer_1_delta => 1 x 4 , how this comes
		# layer_2_delta -> 1x1
		# weights_1_2.T -> 1x4
		layer_1_delta  = layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)

		weights_1_2 -= alpha*(layer_1.T.dot(layer_2_delta)) # 4x1 * 1x1
		weights_0_1 -= alpha*(layer_0.T.dot(layer_1_delta)) # 3x1 * 1x4 -> 3x4

	if iteration % 10 == 9:
		print("Error:" + str(layer_2_error))