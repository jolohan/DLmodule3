import mnist_basics as mb
import tutor1, tutor2, tutor3
import tflowtools as TFT

def test():
	print("Doing a quick test...")
	mb.quicktest()

#test()



# Should hold then ANN's created by the config file:
class NNmodule():

	def __init__(self):

		self.filename = "Config/config.txt"
		self.load_config()

		#tutor3.autoex(epochs=10000,nbits=4,lrate=0.03,showint=100,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,sm=False,bestk=None)
		# Datahandler:
		nbits = 4
		size = 2**nbits
		self.case_generator = self.data_collector

		#print(self.data_collector)
		self.case_manager = tutor3.Caseman(self.case_generator, self.vfrac, self.tfrac)

		# IF not softmax, need apply output activation self:
		self.ann = tutor3.Gann([size, 8, 2], self.case_manager, lrate=self.lr, showint=100, mbs=self.batch_size, vint=100, softmax=(self.activation == 'softmax'))

		# Run:
		self.run()


	def run(self):
		self.ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
		self.ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
		self.ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
		self.ann.run(self.epochs, bestk=False)
		self.ann.runmore(self.epochs, bestk=True)
		string = input("Done? ")

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
		self.hidden_activation = network_dict['HiddenActivation']

		# 3. Output Activation Function
		self.activation = network_dict['OutputActivation']

		# 4. Cost Function (Loss)
		self.cost = 0

		# 5. Learning Rate
		self.lr = float(network_dict['LearningRate'][0])

		# 6. Initial Weight Range (SCALED???)

		# 7. Data Source (data file + function needed to read it) OR (A function name along with parameters)
		source = network_dict['DataSource']

		file_or_function = source[0]
		file = ("." in file_or_function)

		if (file):
			function_name = source[1]
			if (function_name == ""):
				self.data_collector = (lambda : TFT.gen_all_parity_cases(2**nbits))
		else:
			function_name = source[0]
			param = source[1:]

			if (function_name == "load_mnist"):
				self.data_collector = (lambda : mb.load_mnist(param[0]))


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


		# 17. Display Biases (list of the bias vectors to be visualized at the end of the run)


def make_input_output_pairs(img, labels):
	total = int(len(img)*cfrac)
	img, label = img[0:total], labels[0:total]
	cases = [[i, l] for (i, l) in zip(img, label)]
	print("Total cases: ", len(cases))
	return cases

net = NNmodule()