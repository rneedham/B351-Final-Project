# From B351 Final Project (Spring 2018)
# Author: Abe Leite
from tensorflow.python.ops.variables import RefVariable
import tensorflow as tf
import numpy as np
import heapq


def mean_squared_loss(actual, predicted):
    delta = predicted - actual
    return tf.reduce_mean(tf.square(delta))


class Model:
    '''Base Model class, handles most of the boilerplate of training.
    Subclasses are expected to define name, define_vars, define_model, and define_train.'''
    name = ''

    def __init__(self, x_dim, y_dim, loss_factory=None, training=True):
        '''loss_factory returns a TensorFlow function {loss} based on TensorFlow expressions {actual} and {predictions}'''
        self.x_dim = x_dim
        self.y_dim = y_dim

        if loss_factory is None:
            self.loss_factory = mean_squared_loss
        else:
            self.loss_factory = loss_factory

        self.training = training

        if self.training:
            self.x_feed = tf.placeholder(tf.float32, [None, x_dim])
            self.y_feed = tf.placeholder(tf.float32, [None, y_dim])

            self.dataset = tf.data.Dataset.from_tensor_slices((self.x_feed, self.y_feed))
            self.dataset = self.dataset.shuffle(250).batch(100).repeat()
            self.iterator = self.dataset.make_initializable_iterator()

            self.training_x, self.training_y = self.iterator.get_next()

            self.x = tf.placeholder_with_default(self.training_x, [None, x_dim])
            self.correct_y = tf.placeholder_with_default(self.training_y, [None, y_dim])

        else:
            self.x = tf.placeholder(tf.float32, [None, x_dim])
            self.correct_y = tf.placeholder(tf.float32, [None, y_dim])

        prev_dict_keys = set(self.__dict__.keys())
        self.define_variables()
        new_dict_keys = set(self.__dict__.keys())
        var_keys = new_dict_keys - prev_dict_keys

        # prepares to save and load states of the variables defined in self.define_variables()
        self.variables = [self.__dict__[key] for key in sorted(var_keys)
            if type(self.__dict__[key]) == RefVariable]
        
        self.define_model()
        self.loss = self.loss_factory(self.correct_y, self.predicted_y)
        self.optimizer = None
        if self.training:
            self.define_train()

    # General utility functions
    def initialize_training_set(self, session, x_instances, y_instances):
        '''Initialize the training iterator in session with x_instances and y_instances.'''
        session.run(self.iterator.initializer, {self.x_feed: x_instances, self.y_feed: y_instances})

    # Model variable utility functions
    def initialize_variables(self, session):
        '''Initialize vars in session.'''
        session.run(tf.variables_initializer(self.variables))
        if not self.optimizer is None:
            session.run(tf.variables_initializer(self.optimizer.variables()))

    def get_state(self, session):
        '''Returns a state representing the vars in session.'''
        return session.run([var for var in self.variables])

    def set_state(self, session, state):
        '''Sets vars in session according to state.'''
        session.run([var.assign(val) for var, val in zip(self.variables, state)])

    def with_state(self, state):
        '''Generate a partial feed dict that temporarily sets vars according to state.'''
        return {var: val for var, val in zip(self.variables, state)}

    # Validation loss and model application functions
    def check_loss(self, session, x_instances, y_instances, state=None):
        if not state is None:
            feed_dict = self.with_state(state)
        else:
            feed_dict = {}
        feed_dict[self.x] = x_instances
        feed_dict[self.correct_y] = y_instances

        return session.run(self.loss, feed_dict)
    def train_model(self, session, x_instances, y_instances, training_ratio=.8, validation_frequency=1000, validation_threshold=5, validation_patience=50):
        '''Implement the Jack training strategy.

        Arguments:

        session -- tf session object
        x_instances -- values to be provided (nparray of dimension *,self.x_dim)
        y_instances -- values to be predicted (nparray of dimension *,self.y_dim)
        training_ratio -- ratio of instances to include in the training set versus the validation set
        validation_frequency -- number of training steps to perform per validation cycle
        validation_threshold -- number of top states to keep track of at a time
        validation_patience -- number of validation cycles without top-n performance to wait; set to None to wait for KeyboardInterrupt'''

        training_cutoff = int(training_ratio * len(x_instances))

        training_x_instances = x_instances[:training_cutoff]
        training_y_instances = y_instances[:training_cutoff]

        validation_x_instances = x_instances[training_cutoff:]
        validation_y_instances = y_instances[training_cutoff:]

        self.initialize_training_set(session, training_x_instances, training_y_instances)

        best_n = []
        patience = 0
        i = 0
        while True:
            try:
                session.run(self.training_step)
                if i % validation_frequency == 0:
                    val_performace = self.check_loss(session, validation_x_instances, validation_y_instances)

                    report = (-val_performace, self.get_state(session))
                    if len(best_n) < validation_threshold:
                        heapq.heappush(best_n, report)
                    else:
                        heapq.heappushpop(best_n, report)

                    # the logic behind this is to set patience to -1 if
                    # the current report is one of the n best ones.
                    for good_report in best_n:
                        if good_report is report:
                            patience = -1
                            break
                    patience += 1
                    if patience == validation_patience:
                        break

                    print('Validation:', val_performace, end='\t')
                    print('Total:', self.check_loss(session, x_instances, y_instances))
                i += 1
            except KeyboardInterrupt:
                break
        while len(best_n) > 1: heapq.heappop(best_n)

        _, best_state = best_n[0]

        self.set_state(session, best_state)

    def apply_model(self, session, x_instances, state=None):
        if not state is None:
            feed_dict = self.with_state(state)
        else:
            feed_dict = {}
        feed_dict[self.x] = x_instances
        return session.run(self.predicted_y, feed_dict)

    # Model definitions. The meat of the subclass.
    def define_variables(self):
        '''Define vars needed for model.'''
        pass

    def define_model(self):
        '''Defines self.predicted_y based on self.x and any variables.'''
        self.predicted_y = None
        raise NotImplementedError('No implementation for model function.')

    def define_train(self):
        '''Defines self.optimizer and self.training_step based on self.loss.'''
        self.optimizer = None
        self.training_step = None
        raise NotImplementedError('No implementation for training step.')


class LinRegModel(Model): # a sample model defining a linear regression
    name = 'linreg'

    def define_variables(self):
        self.W = tf.Variable(tf.zeros([self.x_dim, self.y_dim]), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([self.y_dim]), dtype=tf.float32)

    def define_model(self):
        self.predicted_y = self.x @ self.W + self.b

    def define_train(self):
        self.optimizer = tf.train.AdamOptimizer(.00007, epsilon=.0001)
        self.training_step = self.optimizer.minimize(self.loss)


def train_model(model_class, x_instances, y_instances, init_state=None):
    '''Instantiates and trains model_class on x_instances and y_instances.
    Returns the trained state of the model (as a list of variables' values).'''

    x_dim = x_instances.shape[1]
    y_dim = y_instances.shape[1]

    with tf.Graph().as_default():
        model = model_class(x_dim, y_dim)

        with tf.Session() as sess:
            model.initialize_variables(sess)

            if not init_state is None: model.set_state(sess, init_state)

            model.train_model(sess, x_instances, y_instances)

            state = model.get_state(sess)
    
    return state

def test_model(model_class, state, x_instances, y_instances):
    '''Instantiates model_class with state and tests it on x_instances and y_instances.
    Returns the loss value over the dataset.
    Should be called on completely different x_instances and y_instances than used
    for training, and only used on the test set immediately before publishing results.'''

    x_dim = x_instances.shape[1]
    y_dim = y_instances.shape[1]
    
    with tf.Graph().as_default():
        model = model_class(x_dim, y_dim)

        with tf.Session() as sess:
            model.initialize_variables(sess)
            model.set_state(sess, state)

            loss = model.check_loss(sess, x_instances, y_instances)

    return loss

def make_predictions(model_class, state, x_instances, y_dim):
    '''Instantiates model_class with state and tests it on x_instances and y_instances.
    Returns the loss value over the dataset.
    Should be called on completely different x_instances and y_instances than used
    for training, and only used on the test set immediately before publishing results.'''

    x_dim = x_instances.shape[1]
    
    with tf.Graph().as_default():
        model = model_class(x_dim, y_dim)

        with tf.Session() as sess:
            model.initialize_variables(sess)
            model.set_state(sess, state)

            predicted_y = model.apply_model(sess, x_instances)

    return predicted_y

def rarityHelper(v):
    if v == b"Free":
       return 0
    elif v == b"Common":
        return 1
    elif v == b"Rare":
        return 2
    elif v == b"Epic":
        return 3
    else:
        return 4
    pass

#takes a model and data set and trains the initalAlpha until it does not produce an alphaDelta change great enough
#returns the alpha reached
def alphaTrainer(model, xVar, yVar, initialAlpha, alphaDeltaTolerance):
    alpha = initialAlpha
    alphaDelta = 5.0
    lossA = 0.0
    lossB = 0.0
    lossDelta = 0.0
    for i in range(10):
        print("step " + str(i))
        if (abs(alphaDelta) < alphaDeltaTolerance):
            break
        
        lossB = lossA
        lossA = test_model(LinRegModel, train_model(LinRegModel, xVar, yVar), xVar, yVar)
        lossDelta = lossA - lossB
        print("loss A: " + str(lossA))
        print("loss B: " + str(lossB))
        print("loss delta: " + str(lossDelta))
        if (lossDelta >= 0.0 and not i == 0):
            alphaDelta = (alphaDelta * -1) / 2
        else:
            alphaDelta = alphaDelta * 2
            
        alpha = alpha * (1 + alphaDelta)
        LinRegModel.optimizer = tf.train.AdamOptimizer(alpha, epsilon=.0001)
        print("alpha: " + str(alpha))
        print("alpha delta: " + str(alphaDelta))
    return alpha

class Card:
    
    def __init__(self, fileName):
        converters = { 5: rarityHelper }
        self.x = np.loadtxt(fileName, delimiter=',', converters = converters, skiprows=1, usecols=(2,3,4,5,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44))
        #This is where the "exploded" 2D array will go
        self.expandedX = []
        for i in range(len(self.x)):
            card = []
            for j in range(len(self.x[i])): #creates inner array of an exploded card
                for k in range(len(self.x[i])):
                    card.append(self.x[i][j] * self.x[i][k])
            card = np.asarray(card)
            self.expandedX.append(card) #appends exploded cards into the new array of exploded cards
        self.expandedX = np.asarray(self.expandedX)
        
        self.y = np.loadtxt(fileName, delimiter=',', skiprows=1, usecols=(49, 50))


trainingCards = Card("classic.csv")
testCards = Card("basic.csv")
trainingState = train_model(LinRegModel, trainingCards.x, trainingCards.y)
np.save("out.npys", trainingState)

#alphaTrainer(LinRegModel, trainingCards.x, trainingCards.y, 0.00007, 0.5)
