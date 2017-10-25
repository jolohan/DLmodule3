import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
import random as rand
import os

# ******* A General Artificial Neural Network ********
# This is the original GANN, which has been improved in the file gann.py

class Gann():

    def __init__(self, dims, cman,lrate=.1,showint=None,mbs=10,vint=None,activation="softmax", loss_func="mse", hidden_activation="relu", init_weights=[-.1, .1], title="ANN"):
        print("initiate config variables")
        self.learning_rate = lrate
        self.title = title
        self.layer_sizes = dims # Sizes of each layer of neurons
        self.show_interval = showint # Frequency of showing grabbed variables
        self.global_training_step = 0 # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = [] # One matplotlib figure for each grabvar
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.dendrogram_viewer = PLT.gca()
        self.caseman = cman
        self.activation = activation
        self.loss_func = loss_func
        self.modules = []
        self.hidden_activation = hidden_activation
        self.init_weights = init_weights
        self.build()

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type,spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self,module_index,type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))
        self.grabvar_figures.append(PLT.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self,module): self.modules.append(module)

    def build(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input; insize = num_inputs
        # Build all of the modules
        for i,outsize in enumerate(self.layer_sizes[1:]):
            if (i < len(self.layer_sizes) - 2):
                gmod = Gannmodule(self,i,invar,insize,outsize,self.hidden_activation, self.init_weights)
                invar = gmod.output; insize = gmod.outsize
            else:
                gmod = Gannmodule(self,i,invar,insize,outsize,"linear", self.init_weights)
                invar = gmod.output; insize = gmod.outsize
        self.output = gmod.output # Output of last module is output of whole network
        if (self.activation == "softmax"):
            self.output = tf.nn.softmax(self.output)
        elif (self.activation == "log_softmax"):
            self.output = tf.nn.log_softmax(self.output)
        elif (self.activation == "sigmoid"):
            self.output = tf.nn.sigmoid(self.output)
        elif (self.activation == "linear"):
            self.output = self.output

        # Sparse softmax cross entropy loss only accepts 1D vectors:
        if (self.loss_func == "sparse_softmax_cross_entropy"):
            self.target = tf.placeholder(tf.int64,shape=(None),name='Target')
        else:
            self.target = tf.placeholder(tf.float64,shape=(None,gmod.outsize),name='Target')
        self.configure_learning()

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self):
        if (self.loss_func == "mse"):
            self.error = tf.reduce_mean(tf.square(self.target - self.output),name='MSE')
        elif (self.loss_func == "L2Loss"):
            self.error = tf.nn.l2_loss(self.output, name="L2Loss")
        elif (self.loss_func == "softmax_cross_entropy"):
            self.error = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.output, name='SoftmaxCrossEntropy')
        elif (self.loss_func == "sparse_softmax_cross_entropy"):
            self.error = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.output, name='SparseSoftmaxCrossEntropy')
        elif (self.loss_func == "sigmoid_cross_entropy"):
            self.error = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=self.output, name='SigmoidCrossEntropy')
        print("ERROR: ", self.error)
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error,name='Backprop')

    def do_training(self,sess,cases,epochs=100,continued=False):
        if not(continued):
            self.error_history = []
        for i in range(epochs):
            error = 0
            step = self.global_training_step + i
            #print("Sending this variable, step =", step)
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size; ncases = len(cases); nmb = math.ceil(ncases/mbs)
            for cstart in range(0,ncases,mbs):  # Loop through cases, one minibatch at a time.
                #print("Training on case: [" + str(cstart) + "/" + str(ncases) + "]")
                cend = min(ncases,cstart+mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]; targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _,grabvals,_ = self.run_one_step([self.trainer],gvars,self.probes,session=sess,
                                         feed_dict=feeder,step=step,show_interval=self.show_interval)
                # Need to account for all errors in a batch:
                batch_error = 0
                for e in grabvals[0]:
                    batch_error += e
                error += float(batch_error/len(grabvals[0]))
            if ((self.validation_interval > 0) and step % self.validation_interval == 0):
                print("\n--- Training ---\n")
                print("Epoch: ", step)
                print('%s Set Error = %f ' % ("Training", float(error/nmb)))
            self.error_history.append((step, error/nmb))
            self.consider_validation_testing(step,sess)
        self.global_training_step += epochs
        TFT.plot_training_history(self.error_history,self.validation_history,xtitle="Epoch",ytitle="Error",
                                  title="",fig=not(continued))

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard MSE error function is used for testing.

    def do_testing(self, sess, cases, msg='Testing', bestk=None, epoch=0):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        #print("\n--- " + msg + " ---\n")
        self.test_func = self.error
        if bestk is not None:
            if (self.loss_func == "sparse_softmax_cross_entropy"):
                self.test_func = self.gen_match_counter(self.predictor, [v for v in targets],k=bestk)
            else:
                self.test_func = self.gen_match_counter(self.predictor,[TFT.one_hot_to_int(list(v)) for v in targets],k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
        if bestk is None:
            if epoch == 0:
                epoch = self.global_training_step
            #print('Epoch: ', epoch)
            if (type(testres) != np.float64 and len(testres) > 1):
                batch_error = 0
                for e in testres:
                    batch_error += e
                batch_error = float(batch_error/len(testres))
                print('%s Set Error = %f ' % (msg, batch_error))
            else:
                print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(testres/len(cases))))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns a OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def do_mapping(self, sess, cases, msg='Mapping', bestk=None):
        print("do mapping")
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        print(targets)
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.predictor
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor,[TFT.one_hot_to_int(list(v)) for v in targets],k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                                 feed_dict=feeder, show_interval=1)
        if bestk is None:
            print("Error: ", testres)
            # print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100 * (testres / len(cases))))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    def make_dendogram(self, sess, cases, first, msg='Dendogram', bestk=None):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        target_dict = {}
        for c in range(len(inputs)):
            feeder = {self.input: [inputs[c]], self.target: [targets[c]]}
            self.test_func = self.predictor
            if bestk is not None:
                self.test_func = self.gen_match_counter(self.predictor,targets[c],k=bestk)
            testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                                     feed_dict=feeder, show_interval=1)
            activation = [i for i in testres[0]]
            if (type(targets[c]) == int):
                out_string = str(targets[c])
            else:
                out_string = TFT.bits_to_str(targets[c])
            if (not out_string in target_dict):
                target_dict[out_string] = activation

        activations = []
        labels = []
        for key in target_dict.keys():
            labels.append(key)
            activations.append(target_dict[key])
        title="Dendrogram_" + self.title
        if first:
            title += "_early"
        TFT.dendrogram(features=activations, labels=labels, ax=self.dendrogram_viewer, title=title)

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        self.roundup_probes()
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session,self.caseman.get_training_cases(),epochs,continued=continued)

    def testing_session(self,sess,bestk=None):
        print("\n---Testing Session---\n")
        cases = self.caseman.get_testing_cases()
        print("Cases: ", len(cases))
        if len(cases) > 0:
            self.do_testing(sess,cases,msg='Final Testing',bestk=bestk)

    def consider_validation_testing(self,epoch,sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            test_cases = self.caseman.get_testing_cases()
            if len(cases) > 0:
                self.make_dendogram(sess, cases, (epoch==self.validation_interval))
                error = self.do_testing(sess,cases,msg='Validation Testing',epoch=epoch)
                self.validation_history.append((epoch,error))
            if (len(test_cases) > 0):
                error = self.do_testing(sess, test_cases, msg='Testset Testing', epoch=epoch)

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess,bestk=None):
        self.do_testing(sess,self.caseman.get_training_cases(),msg='Total Training',bestk=bestk)

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=0):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        #if show_interval and (step % show_interval == 0) and step > 0:
        if (show_interval==1):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars,step=1):
        names = [x.name for x in grabbed_vars];
        msg = "Grabbed Variables at Step " + str(step)
        #print("\n" + msg, end="\n")
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                fig = self.grabvar_figures[fig_index]
                if fig == None:
                    print('FIGURE IS NONE')
                print("hint plot fig: "+str(fig_index))
                TFT.hinton_plot(v,fig=fig,title= names[i]+ ' at step '+ str(step))
                fig_index += 1

    def run(self,epochs=100,sess=None,continued=False,bestk=None):
        PLT.ion()
        self.training_session(epochs,sess=sess,continued=continued)
        self.test_on_trains(sess=self.current_session,bestk=bestk)
        self.testing_session(sess=self.current_session,bestk=bestk)
        self.close_current_session(view=False)
        PLT.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self,epochs=100,bestk=None):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True,bestk=bestk)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        directory = os.path.dirname(spath)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self,view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule():

    def __init__(self,ann,index,invariable,insize,outsize,activation,weights):
        self.ann = ann
        self.insize=insize  # Number of neurons feeding into this module
        self.outsize=outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-"+str(self.index)
        self.activation = activation
        self.weights = weights
        self.build()

    def build(self):
        mona = self.name; n = self.outsize
        self.weights = tf.Variable(np.random.uniform(self.weights[0], self.weights[1], size=(self.insize,n)),
                                   name=mona+'-wgt',trainable=True) # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n),
                                  name=mona+'-bias', trainable=True)  # First bias vector
        print(self.activation)
        if (self.activation == "relu"):
            self.output = tf.nn.relu(tf.matmul(self.input,self.weights)+self.biases,name=mona+'-out')
        elif (self.activation == "sigmoid"):
            self.output = tf.nn.sigmoid(tf.matmul(self.input,self.weights)+self.biases,name=mona+'-out')
        elif (self.activation == "linear"):
            self.output = tf.matmul(self.input,self.weights)+self.biases


        self.ann.add_module(self)

    def getvar(self,type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self,type,spec):
        var = self.getvar(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)

# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman():

    def __init__(self,cfunc,vfrac=0,tfrac=0, cfrac=1.0):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.case_fraction = cfrac
        print("casefraction = "+str(self.case_fraction))
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    # Dealing with both shuffle and case-fraction her, so not the same cases are removed everytime we restart...
    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]
        rand.shuffle(self.cases)
        total_cases = int(self.case_fraction*len(self.cases))
        self.cases = self.cases[:total_cases]
        print("Total cases: ", len(self.cases))

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca) # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases


#   ****  MAIN functions ****

# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.
def autoex(epochs=300,nbits=4,lrate=0.03,showint=100,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,sm=False,bestk=None):
    size = 2**nbits
    mbs = mbs if mbs else size
    case_generator = (lambda : TFT.gen_all_one_hot_cases(2**nbits))
    cman = Caseman(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    ann = Gann(dims=[size,nbits*8,size],cman=cman,lrate=lrate,showint=showint,mbs=mbs,vint=vint,softmax=sm)
    ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs, bestk=True)
    ann.runmore(epochs, bestk=True)
    return ann

def countex(epochs=5000,nbits=10,ncases=500,lrate=0.5,showint=500,mbs=20,vfrac=0.1,tfrac=0.1,cfrac=1.0,vint=200,sm=True,bestk=1):
    case_generator = (lambda: TFT.gen_vector_count_cases(ncases,nbits))
    cman = Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac, cfrac=cfrac)
    ann = Gann(dims=[nbits, nbits*3, nbits+1], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint, softmax=sm)
    ann.run(epochs,bestk=bestk)
    return ann

