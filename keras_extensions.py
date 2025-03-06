'''
Keras is great, but it makes certain assumptions that do not quite work for NLP problems. We override some of those
assumptions here.
'''
# from overrides import overrides

from keras import backend as K
from keras.layers import Embedding, TimeDistributed, Flatten
from keras import ops as keras_ops

if K.backend() == 'torch':
    import torch as TORCH
    KBEND = TORCH
elif K.backend() == 'theano':
    import theano.tensor as T
    KBEND = T
else:
    import tensorflow as TF
    KBEND = TF

class AnyShapeEmbedding(Embedding):
    '''
    We just want Embedding to work with inputs of any number of dimensions.
    This can be accomplished by simply changing the output shape computation.
    '''
    # @overrides
    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)


class TimeDistributedRNN(TimeDistributed):
    '''
    The TimeDistributed wrapper in Keras works for recurrent layers as well, except that it does not handle masking
    correctly. In case when the wrapper recurrent layer does not return a sequence, no mask is returned. However,
    when we are time distributing it, it is possible that some sequences are entirely padding, for example, when
    one of the slots being encoded is not present in the input at all. We override masking here.
    '''
    # @overrides
    def compute_mask(self, x, input_mask=None):
        # pylint: disable=unused-argument
        if input_mask is None:
            return None
        else:
            return keras_ops.any(input_mask, axis=-1)


class MaskedFlatten(Flatten):
    '''
    Flatten does not allow masked inputs. This class does.
    '''
    def __init__(self, **kwargs):
        super(MaskedFlatten, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # Assuming the output will be passed through a dense layer after this.
        if mask is not None:
            inputs = __switch__(keras_ops.expand_dims(mask), inputs, keras_ops.zeros_like(inputs))
        return super(MaskedFlatten, self).call(inputs)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            if keras_ops.ndim(mask) == 2:
                # This needs special treatment. It means that the input ndim is 3, and output ndim is 2, thus
                # requiring the mask's ndim to be 1.
                return keras_ops.any(mask, axis=-1)
            else:
                return __flatten__(mask)


def __flatten__(input):
    '''
    Keras' implementation of flatten does not work with masked inputs. This function selects the appropriate methods.
    '''
    if K.backend() == 'torch':
        return KBEND.flatten(input)
    else:
        return KBEND.batch_flatten(input)


def __switch__(cond, then_tensor, else_tensor):
    '''
    Keras' implementation of switch for tensorflow works differently compared to that for torch and theano. This function
    selects the appropriate methods.
    '''
    if K.backend() == 'torch':
        return KBEND.where(cond, then_tensor, else_tensor)
    elif K.backend() == 'tensorflow':
        tf = KBEND
        cond_shape = cond.get_shape()
        input_shape = then_tensor.get_shape()
        if cond_shape[-1] != input_shape[-1] and cond_shape[-1] == 1:
            # This means the last dim is an embedding dimension.
            cond = keras_ops.dot(tf.cast(cond, tf.float32), tf.ones((1, input_shape[-1])))
        return tf.where(tf.cast(cond, dtype=tf.bool), then_tensor, else_tensor)
    else:
        return KBEND.switch(cond, then_tensor, else_tensor)
