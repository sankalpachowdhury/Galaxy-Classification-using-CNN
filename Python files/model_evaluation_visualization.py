#from tensorflow.keras.models import load_model

# model_param_file = 'best_model.h5'
# val_gen = validation_generator
# train_gen = train_generator

# model evaluation
def model_evaluation(model_param_file, train_gen, val_gen):
    # loading weights and bisased from the stored files
    saved_model = load_model(model_param_file)
    _, train_acc = saved_model.evaluate(train_gen)
    _, validation_acc = saved_model.evaluate(val_gen)
    print('\nTraining accuracy: {0}\nValidation accuracy: {1}'.format(train_acc, validation_acc))
    print('\nValidation output:\n',model.predict_classes(val_gen))

# model training visualization curve
def train_visualization(history):
    fig = plt.figure(figsize=(14, 10))
    fig.add_subplot(1, 1, 1)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))  # range for the number of epochs
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy for the model')
