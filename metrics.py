'''
We define some custom Keras metrics here that are specific to binary classification.
'''

from keras import backend as K
from keras import ops as keras_ops


def precision(y_true, y_pred):
    '''
    Custom Keras metric that measures the precision of a binary classifier.
    '''
    # Assuming index 1 is positive.
    pred_indices = keras_ops.argmax(y_pred, axis=-1)
    true_indices = keras_ops.argmax(y_true, axis=-1)
    num_true_positives = keras_ops.sum(pred_indices * true_indices)
    num_positive_predictions = keras_ops.sum(pred_indices)
    return keras_ops.cast(num_true_positives / num_positive_predictions, K.floatx())


def recall(y_true, y_pred):
    '''
    Custom Keras metric that measures the recall of a binary classifier.
    '''
    # Assuming index 1 is positive.
    pred_indices = keras_ops.argmax(y_pred, axis=-1)
    true_indices = keras_ops.argmax(y_true, axis=-1)
    num_true_positives = keras_ops.sum(pred_indices * true_indices)
    num_positive_truths = keras_ops.sum(true_indices)
    return keras_ops.cast(num_true_positives / num_positive_truths, K.floatx())


def f1_score(y_true, y_pred):
    '''
    Custom Keras metric that measures F1 score of a binary classifier.
    '''
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return keras_ops.cast(2 * prec * rec / (prec + rec), K.floatx())
