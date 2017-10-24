import mnist_basics as mb
import tutor1, tutor2, tutor3
import tflowtools as TFT
from numpy import linalg as la

def test():
	print("Doing a quick test...")
	mb.quicktest()

#test()



# Should hold the ANN's created by the config file:
class NNmodule():

	def __init__(self):

		# Load config file:
		self.filename = "Config/seg_count_config.txt"
		self.load_config()

		# Datahandler (sent to CASEMAN):
		self.case_generator = self.data_collector

		# CASEMAN itself:
		self.case_manager = tutor3.Caseman(self.case_generator, self.vfrac, self.tfrac, self.cfrac)

		# Artificial Neural Network:
		self.ann = tutor3.Gann(self.sizes, self.case_manager, lrate=self.lr, showint=None, mbs=self.batch_size, vint=1,
			activation=self.activation, loss_func=self.loss_func, hidden_activation=self.hidden_activation, init_weights=self.weight_range)

		# Run the network:
		self.run()


	def run(self):
		self.ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
		if (len(self.sizes) > 2):
			self.ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
		self.ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
		self.ann.run(self.epochs, bestk=True)
		done = False
		while not done:
			string = input("Do you want to run more epochs? [0, n]\n: ")
			try:
				more_epochs = int(string)
				if (more_epochs > 0):
					self.ann.runmore(more_epochs, bestk=True)
				else:
					done = True
			except:
				done = True
				print("Shutting down...")
		print("Shutting down...")

	def load_config(self):

		network_dict = {}

		# Pre-processing config file:
		with open(self.filename, "r") as file:

			for line in file:
				
				listed_specs = line.split(" ")

				network_dict[listed_specs[0]] = [item.strip() for item in listed_specs[1:]]

		print("Dictionary: ", network_dict)

		# 1. Network Dimensions (+ sizes)
		sizes = network_dict['NetSize']
		self.sizes = []
		
		for s in sizes:
			self.sizes.append(int(s))

		# 2. Hidden Activation Function
		self.hidden_activation = network_dict['HiddenActivation'][0]

		# 3. Output Activation Function
		self.activation = network_dict['OutputActivation'][0]

		# 4. Cost Function (Loss)
		self.loss_func = network_dict['LossFunc'][0]

		# 5. Learning Rate
		self.lr = float(network_dict['LearningRate'][0])

		# 6. Initial Weight Range
		self.weight_range = [float(i) for i in network_dict['InitialWeight']]

		# 7. Data Source (data file + function needed to read it) OR (A function name along with parameters)
		source = network_dict['DataSource']

		file_or_function = source[0]
		file = ("." in file_or_function)

		if (file):
			function_name = source[1]
			if (function_name == "gen_all_parity_cases"):
				self.data_collector = (lambda : TFT.gen_all_parity_cases(nbits))
		else:
			function_name = source[0]
			params = source[1:]
			self.data_collector = (lambda : manage_data_loaders(function_name, params, self.loss_func))



		# 8. Case Fraction (Std: 1.0, how much of the original dataset to use)
		self.cfrac = float(network_dict['CaseFrac'][0])

		# 9. Validation Fraction
		self.vfrac = float(network_dict['ValFrac'][0])

		# 10. Test Fraction
		self.tfrac = float(network_dict['TestFrac'][0])

		# 11. Minibatch Size
		self.batch_size = int(network_dict['BatchSize'][0])

		# 12. Map Batch Size (0 = no Map test)
		self.map_batch_size = int(network_dict['MapBatchSize'][0])

		# 13. Steps (Total number of MINIBATCHES during training)
		self.epochs = int(network_dict['Epochs'][0])

		# 14. Map Layers (The layers to be visualized during the map test)
		if (self.map_batch_size > 0):
			self.map_layers = [int(layer) for layer in network_dict['MapLayers']]

			
		# 15. Map Dendograms (List of layers whose activation patterns will be used to produce Dendograms (Map Test))
		if (self.map_batch_size > 0):
			self.map_dendos = [int(layer) for layer in network_dict['MapDendo']]

		# 16. Display Weights (list of the weight arrays to be visualized at the end of the run)
		self.display_weights = [int(layer) for layer in network_dict['DisplayWeights']]

		# 17. Display Biases (list of the bias vectors to be visualized at the end of the run)
		self.display_biases = [int(layer) for layer in network_dict['DisplayBias']]

# For the MNIST dataset:
def make_input_output_pairs(parameters, loss_function):
	dataset = 0
	digits = []
	for p in parameters:
		if (p=="testing"):
			dataset = 1
		if (p.isdigit()):
			digits.append(int(p))
	output_size = 10
	if (len(digits) == 0):
		images, labels = mb.load_mnist(dataset=("training" if not dataset else "testing"))

	# If specified which images to get: (i.e. [1, 4, 6])
	else:
		images, labels = mb.load_mnist(dataset=("training" if not dataset else "testing"), digits=digits)

	# Creating [input, output] - cases, with normalized, flattened images, and int label vectors as output (sparse need integers, not vectors):
	if (loss_function == "sparse_softmax_cross_entropy"):
		cases = [[mb.flatten_image(i)/la.norm(i), int(l[0])] for (i, l) in zip(images, labels)]
		print(cases[0])
	# Creating [input, output] - cases, with normalized, flattened images, and one-hot vectors as output:
	else:
		cases = [[mb.flatten_image(i)/la.norm(i), TFT.int_to_one_hot(int(l[0]), output_size)] for (i, l) in zip(images, labels)]
	print("Total cases collected: ", len(cases))
	return cases


def manage_data_loaders(function_name, params, loss_function):
	one_hot = False
	if (function_name == "load_mnist"):
		cases = make_input_output_pairs(params, self.loss_func)

	elif (function_name == "gen_all_parity_cases"):
		if (len(params) == 1):
			cases = TFT.gen_all_parity_cases(int(params[0]))
		else:
			cases = TFT.gen_all_parity_cases(int(params[0]), bool(params[1]))

	elif (function_name == "gen_all_one_hot_cases"):
		one_hot = True
		cases = TFT.gen_all_one_hot_cases(2**int(params[0]))

	elif (function_name == "gen_vector_count_cases"):
		one_hot = True
		nof_cases = int(params[0])
		length = int(params[1])
		cases = TFT.gen_vector_count_cases(nof_cases, length)

	elif (function_name == "gen_segmented_vector_cases"):
		one_hot = True
		# ASSUMING all these parameters are given:
		try:
			nbits = int(params[0])
			nof_cases = int(params[1])
			min_seg = int(params[2])
			max_seg = int(params[3])
		except:
			print("Not enough arguments given! [nbits, nof_cases, min_seg, max_seg]")
			return
		cases = TFT.gen_segmented_vector_cases(nbits, nof_cases, min_seg, max_seg)


	if (loss_function == "sparse_softmax_cross_entropy"):
		bit_cases = []
		for case in cases:
			if (function_name == "gen_all_parity_cases"):
				if (len(case[1]) == 1):
					bit_cases.append([case[0], case[1]])
				else:
					bit_cases.append([case[0], case[1][1]])
			else:
				for i, b in enumerate(case[1]):
					print(case[1])
					if (b):
						bit_cases.append([case[0], i])
						break
		cases = bit_cases
	print(cases[0:])
	return cases




net = NNmodule()