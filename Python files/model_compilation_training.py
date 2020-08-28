from modules import *

# from tensorflow.keras.optimizers import Adam 
# loss_func = 'categorical_crossentropy'
# metrices = ['acc']
# optimizer_func = Adam(learning_rate=0.001

# model compilation
def model_compile(model, optimizer_func, loss_func, metrices):
    model.compile(optimizer = optimizer_func,
              loss = loss_func,
              metrics = metrices)

# threshold_loss = 0.2000
# model_param_file = 'best_model.h5'
# monitor_mc = 'val_accuracy'
# mode_mc = 'max'

# custom callback 
def model_callbacks(threshold_loss, model_param_file, monitor_mc, mode_mc):
    # custom callback
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = {}):
            if(logs.get('val_loss') < threshold_loss):
                self.model.stop_training = True

    # instance for myCallback
    call_stop = myCallback()

    # earlystopping(Not used here in training)
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose=1, patience = 50, baseline = 0.2791)

    # ModelCheckpoint object
    mc = ModelCheckpoint(model_param_file, monitor = monitor_mc, mode = mode_mc, verbose = 0)

    return call_stop, es, mc

# train_gen = train_generator
# val_gen = validation_generator
# epoch_count = EPOCHS

# model training
def model_training(model, train_gen, val_gen, epoch_count, call_stop, es, mc):
    history = model.fit_generator(train_gen,
                    epochs = epoch_count,
                    validation_data = val_gen,
                    callbacks=[call_stop, mc],
                    verbose=1)
    return history
