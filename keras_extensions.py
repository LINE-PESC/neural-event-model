'''
Keras is great, but it makes certain assumptions that do not quite work for NLP problems. We override some of those
assumptions here.
'''
#from overrides import overrides

from keras import backend as K
from keras.layers import Embedding, TimeDistributed, Flatten, Reshape, RepeatVector


class AnyShapeEmbedding(Embedding):
    '''
    We just want Embedding to work with inputs of any number of dimensions.
    This can be accomplished by simply changing the output shape computation.
    '''
    #@overrides
    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)


class TimeDistributedRNN(TimeDistributed):
    '''
    The TimeDistributed wrapper in Keras works for recurrent layers as well, except that it does not handle masking
    correctly. In case when the wrapper recurrent layer does not return a sequence, no mask is returned. However,
    when we are time distributing it, it is possible that some sequences are entirely padding, for example, when
    one of the slots being encoded is not present in the input at all. We override masking here.
    '''
    #@overrides
    def compute_mask(self, x, input_mask=None):
        # pylint: disable=unused-argument
        if input_mask is None:
            return None
        else:
            return K.any(input_mask, axis=-1)


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
            inputs = switch(K.expand_dims(mask), inputs, K.zeros_like(inputs))
        return super(MaskedFlatten, self).call(inputs)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            if K.ndim(mask) == 2:
                # This needs special treatment. It means that the input ndim is 3, and output ndim is 2, thus
                # requiring the mask's ndim to be 1.
                return K.any(mask, axis=-1)
            else:
                return K.batch_flatten(mask)


class MaskedRepeat(RepeatVector):
    '''
    RepeatVector does not allow masked inputs and does not work with inputs of any number of dimensions. This class does.
    '''
    def __init__(self, n, axis=1, **kwargs):
        super(MaskedRepeat, self).__init__(n, **kwargs)
        self.axis = axis
        self.input_spec = None
        self.supports_masking = True
 
    #@overrides
    def compute_output_shape(self, input_shape):
        if (len(input_shape) <= self.axis):
            raise ValueError('`axis` value can not bigger than the number of dimensions from input_shape.')
        return (input_shape[:self.axis]) + ((input_shape[self.axis] * self.n),) + (input_shape[self.axis+1:])
      
    #@overrides
    def call(self, inputs):
        return K.repeat_elements(inputs, self.n, self.axis)
    
# TODO: precisa checar se com o suporte a masl_zero, a camada está realmente funcionando como deveria
#     def compute_mask(self, inputs, mask=None):
#         if mask is None:
#             return None
#         else:
#             if K.ndim(mask) == 2:
#                 # This needs special treatment. It means that the input ndim is 3, and output ndim is 2, thus
#                 # requiring the mask's ndim to be 1.
#                 return K.any(mask, axis=-1)
#             else:
#                 return K.batch_flatten(mask)
 
 
class MaskedReshape(Reshape):
    '''
    Reshape does not allow masked inputs. This class does.
    '''
    def __init__(self, target_shape, **kwargs):
        super(MaskedReshape, self).__init__(target_shape, **kwargs)
        self.supports_masking = True
 
# TODO: precisa checar se com o suporte a masl_zero, a camada está realmente funcionando como deveria
#     def call(self, inputs, mask=None):
#         # Assuming the output will be passed through a dense layer after this.
#         if mask is not None:
#             inputs = switch(K.expand_dims(mask), inputs, K.zeros_like(inputs))
#         return super(MaskedReshape, self).call(inputs)
#  
# TODO: precisa checar se com o suporte a masl_zero, a camada está realmente funcionando como deveria
#     def compute_mask(self, inputs, mask=None):
#         if mask is None:
#             return None
#         else:
#             if K.ndim(mask) == 2:
#                 # This needs special treatment. It means that the input ndim is 3, and output ndim is 2, thus
#                 # requiring the mask's ndim to be 1.
#                 return K.any(mask, axis=-1)
#             else:
#                 return K.batch_flatten(mask)


def switch(cond, then_tensor, else_tensor):
    '''
    Keras' implementation of switch for tensorflow works differently compared to that for theano. This function
    selects the appropriate methods.
    '''
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        cond_shape = cond.get_shape()
        input_shape = then_tensor.get_shape()
        if cond_shape[-1] != input_shape[-1] and cond_shape[-1] == 1:
            # This means the last dim is an embedding dimension.
            cond = K.dot(tf.cast(cond, tf.float32), tf.ones((1, input_shape[-1])))
        return tf.where(tf.cast(cond, dtype=tf.bool), then_tensor, else_tensor)
    else:
        import theano.tensor as T
        return T.switch(cond, then_tensor, else_tensor)
