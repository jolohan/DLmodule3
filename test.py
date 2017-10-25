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

	def __init__(self, config):

		# Load config file:
		print("config: ", config)
		self.filename = config
		self.load_config()

		# Datahandler (sent to CASEMAN):
		self.case_generator = self.data_collector

		# CASEMAN itself:
		self.case_manager = tutor3.Caseman(self.case_generator, self.vfrac, self.tfrac, self.cfrac)

		# Artificial Neural Network:
		self.ann = tutor3.Gann(self.sizes, self.case_manager, lrate=self.lr, showint=self.showint, mbs=self.batch_size, vint=self.vint,
			activation=self.activation, loss_func=self.loss_func, hidden_activation=self.hidden_activation, init_weights=self.weight_range, title=self.filename[7:len(self.filename)-4])

		# Run the network:
		self.run()

	def add_weights_and_biases_to_display(self):
		if (len(self.display_weights) > 0):
			for layer in self.display_weights:
				self.ann.add_grabvar(layer, 'wgt')
		if (len(self.display_biases) > 0):
			for layer in self.display_biases:
				self.ann.add_grabvar(layer, 'bias')

	def run(self):
		self.ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
		if (len(self.sizes) > 2):
			self.ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
		self.add_weights_and_biases_to_display()
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
		self.ann.grabvars = []
		self.ann.grabvar_figures = []
		if (self.map_batch_size > 0):
			if self.map_layers[0] == 0:
				self.ann.add_grabvar(0, 'in')
			for layer in self.map_layers:
				self.ann.add_grabvar(layer, 'out')
		#print("Grabbed vars: ")
		#for var in self.ann.grabvars:
		#	print(var)
		if (self.map_batch_size > 0):
			sess = self.ann.reopen_current_session()
			cases = self.ann.caseman.get_testing_cases()[0:self.map_batch_size]
			self.ann.do_mapping(sess, cases=cases)
			#############################
		print("Shutting down...")

	def load_config(self):

		network_dict = {}

		# Pre-processing config file:
		with open(self.filename, "r") as file:

			for line in file:
				
				listed_specs = line.split(" ")

				network_dict[listed_specs[0]] = [item.strip() for item in listed_specs[1:]]

		print("Dictionary: ", network_dict)

		# Parameters for output generation:
		self.vint = network_dict['VInt'][0]
		if (self.vint == "None" or self.vint == "0"):
			self.vint = None
		else:
			self.vint = int(self.vint)

		self.showint = network_dict['ShowInt'][0]
		if (self.showint == "None" or self.showint == "0"):
			self.showint = None
		else:
			self.showint = int(self.showint)

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
			filename = file_or_function
			function_name = source[1]
			self.data_collector = (lambda : manage_data_loaders(function_name, [filename], self.loss_func))
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
def mnist(parameters, loss_function):
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
	cases = [[mb.flatten_image(i)/la.norm(i), TFT.int_to_one_hot(int(l[0]), output_size)] for (i, l) in zip(images, labels)]
	print("Total cases collected: ", len(cases))
	return cases

# For loading the UC Irvine datasets:
def dataset_loader(filename, loss_function):
	print(filename)
	with open(filename, "r") as file:
		feature_vectors = []
		labels = []
		splitter = ";"
		for line in file:
			if (len(line) > 0):
				if (splitter not in line):
					splitter = ","
				split_string = line.split(splitter)
				labels.append(int(split_string[-1]))
				feature_vectors.append([float(i) for i in split_string[:len(split_string) - 1]])
	print("Nof features: ", len(feature_vectors[0]))
	print("Nof examples: ", len(feature_vectors))
	print(max(labels), min(labels))

	# Making one-hot-labels:
	one_hot_labels = [TFT.int_to_one_hot(l, max(labels) + 1)[min(labels):] for l in labels]
	# Normalizing features in the space [0, 1]:
	normalized_feature_vectors = normalize_features(feature_vectors)

	# Creating the case-set:
	cases = [[f, l] for (f, l) in zip(normalized_feature_vectors, one_hot_labels)]
	return cases

def normalize_features(vector):
	nof_features = len(vector[0])
	f_max = []
	f_min = []
	for i in range(nof_features):
		for j in range(len(vector)):
			cell = vector[j][i]
			if (len(f_max) <= i):
				f_max.append(cell)
			else:
				if (cell > f_max[i]):
					f_max[i] = cell
			if (len(f_min) <= i):
				f_min.append(cell)
			else:
				if (cell < f_min[i]):
					f_min[i] = cell

	print(f_min)
	print(f_max)

	normalized_features = []
	for c, case in enumerate(vector):
		normalized_features.append([(f_c_j-f_min[j])/(f_max[j]-f_min[j]) if f_max[j]-f_min[j] != 0 else 1 for j, f_c_j in enumerate(case)])

	#print(normalized_features[0:10])
	return normalized_features


def manage_data_loaders(function_name, params, loss_function):
	one_hot = False
	if (function_name == "load_mnist"):
		cases = mnist(params, loss_function)

	elif (function_name == "dataset_loader"):
		cases = dataset_loader(params[0], loss_function)

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


	# Sparse need integers as labels, not vectors:
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
					if (b):
						bit_cases.append([case[0], i])
						break
		cases = bit_cases
	return cases



if __name__ == '__main__':

	print("\n--- ANN Module Interface ---\n")

	config_dictionary = {0: 'parity_config.txt',
	1: 'bit_count_config.txt',
	2: 'seg_count_config.txt',
	3: 'mnist_config.txt',
	4: 'red_wine_config.txt',
	5: 'yeast_config.txt',
	6: 'glass_config.txt',
	7: 'config.txt',
	8: 'Exit'}

	finished = False

	while not finished:
		for key in config_dictionary.keys():
			print(str(key) + ": ", config_dictionary[key])

		config = input("\nWhich config to run [0/" + str(len(config_dictionary)-1) + "]: ")
		config_nr = int(config)

		if (config_nr == 8):
			finished = True
			break

		net = NNmodule("Config/" + config_dictionary[config_nr])

		done = input("Exit = 0, continue = 1: ")

		try:
			done = int(done)
			if (not done):
				finished = True
				break

		except:
			print("Continuing...")

	print("Exiting...")