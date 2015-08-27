from lasagne.layers import Layer
from lasagne import init
from lasagne import nonlinearities
import theano.tensor as T


class RNNDenseLayer(Layer):
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        """A convenience DenseLayer that cooperates with recurrent layers.

        Recurrent layers work on 3-dimensional data (batch size x time
        x number of units). By default, Lasagne DenseLayer flattens
        data to 2 dimensions. We could reshape the data or we could
        just use this RNNDenseLayer, which is more convenient.

        For documentation, refer to Lasagne's DenseLayer documenation.
        """
        super(RNNDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = self.input_shape[2]

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim != 3:
            raise ValueError("Need 3 dimensional input, man.")

        # bs x time x num_tokens
        shape = input.shape
        # bs * time x num_tokens
        input = T.reshape(input, (shape[0] * shape[1], shape[2]))

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        output = self.nonlinearity(activation)

        # bs x time x num_tokens
        output = T.reshape(
            self.nonlinearity(output),
            (shape[0], shape[1], self.num_units))
        return output
