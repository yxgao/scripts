import os.path
import sys

from keras.models import Sequential
from keras.seq2seq.models import Seq2seq
from utils.utils import get_logger

_logger = get_logger(__name__)

"""
build model function
"""
def get_SeqAE_model(batch_input_shape=None,
                hidden_dim=None,
                output_dim=None,
                output_length=None,
                depth=1,
                peek=False,
                loss='mse',
                optimizer='rmsprop',
                saved_path=None):
    
    model = Sequential()
    seq2seq = Seq2seq(
                batch_input_shape=batch_input_shape,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                output_length=output_length,
                depth=depth,
                peek=peek
    )

    model.add(seq2seq)
    model.compile(loss=loss, optimizer=optimizer)

    # use previously saved model if it exists
    _logger.info('Looking for a model %s' % saved_path)

    if os.path.isfile(saved_path):
        _logger.info('Loading previously calculated weights...')
        model.load_weights(saved_path)

    _logger.info('Model is built')
    return model



"""
train model functions
"""
def train_SeqAE_model(data,
                      nn_model,
                      epochs,
                      learning_rate,
                      learning_decay):

    start_time = time.time()
    sents_batch_iteration = 1

    for full_data_pass_num in xrange(1, FULL_LEARN_ITER_NUM + 1):
        _logger.info('Full-data-pass iteration num: ' + str(full_data_pass_num))
        dialog_lines_for_train = copy.copy(tokenized_dialog_lines)

        for X_train, Y_train in get_training_batch():
            nn_model.fit(X_train, 
                         Y_train, 
                         batch_size=TRAIN_BATCH_SIZE, 
                         nb_epoch=1, 
                         show_accuracy=True, 
                         verbose=1)

            if sents_batch_iteration % TEST_PREDICTIONS_FREQUENCY == 0:
                log_predictions(test_sentences, nn_model, w2v_model, index_to_token)
                save_model(nn_model)

            sents_batch_iteration += 1

        _logger.info('Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / full_data_pass_num))
    save_model(nn_model)

"""
using model to predict
"""