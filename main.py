from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from net_architecture import CRNN_model, CRNN_MODE
from net_config import ArchitectureConfig
from utils import load_data, TextSequenceGenerator, decode_predict_ctc, labels_to_text
from net_config import FilePaths
import numpy as np
import enum
import os



def train(start = 0, end = 1000, no_epochs = ArchitectureConfig.EPOCHS):
    data = load_data(start, end)
    no_samples = len(data)
    no_train_set = int(no_samples * 0.95)
    no_val_set = no_samples - no_train_set

    train_set = TextSequenceGenerator(data[:no_train_set])
    test_set = TextSequenceGenerator(data[no_train_set:])

    model, y_func = CRNN_model(CRNN_MODE.training)
    if os.path.isfile(FilePaths.fnSave):
        model.load_weights(FilePaths.fnSave)

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    ckp = ModelCheckpoint(
        FilePaths.fnSave,
        monitor='val_loss',
        verbose=1, save_best_only=True, save_weights_only=True
    )
    earlystop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'
    )

    model.fit_generator(generator=train_set,
                        steps_per_epoch=no_train_set // ArchitectureConfig.BATCH_SIZE,
                        epochs=no_epochs,
                        validation_data=test_set,
                        validation_steps=no_val_set // ArchitectureConfig.BATCH_SIZE,
                        callbacks=[ckp, earlystop])

    return model, y_func

def predict(index_batch, index_img):
    data = load_data(start = 0, end = 1000)
    no_samples = len(data)
    no_train_set = int(no_samples * 0.95)
    no_val_set = no_samples - no_train_set

    test_set = TextSequenceGenerator(data[no_train_set:])

    model = CRNN_model(CRNN_MODE.inference)
    model.load_weights(FilePaths.fnSave)

    samples = test_set[index_batch]
    img = samples[0]['the_input'][index_img]
    chars_ = ArchitectureConfig.CHARS

    # plt.imshow(np.squeeze(img).T)
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    net_out_value = model.predict(img)
    print(net_out_value.shape)
    pred_texts = top_pred_texts = decode_predict_ctc(net_out_value, chars_)
    print(pred_texts)
    gt_texts = test_set[index_batch][0]['the_labels'][index_img]
    gt_texts = labels_to_text(chars_, gt_texts.astype(int))
    print(gt_texts)

def main_train():
    no_epochs = 15
    model, y_func = train(start = 0, end = 35, no_epochs = no_epochs)

    model.save_weights(FilePaths.fnSave)

def main_infer():
    predict(1, 0)
    predict(1, 1)

if __name__ == "__main__":
    #predict(1, 0)

    main_train()