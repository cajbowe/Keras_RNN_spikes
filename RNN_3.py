import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

spike_data = np.load('/home/mouse/Downloads/binned_spikes_50ms.npy')

units = 128
dropout = 0.3
learn_rate = 0.002
batch = 64
#normalizing the data
mean, std = spike_data.mean(axis=0), spike_data.std(axis=0)
neural_data = (spike_data - mean) / std
#Setting variables for the 500ms section of spikes used
example_length = 10 #500ms of 50ms bins
example_width = spike_data.shape[1] #Number of active neurons
#Creating the architecture for the model
model = keras.Sequential()
model.add(layers.LSTM(units, input_shape=((example_length-1), example_width), activation='relu', recurrent_dropout=0.1))
model.add(layers.Dropout(dropout, input_shape=((example_length-1), example_width)))
model.add(layers.Dense(example_width))
print(model.summary())
#Defining the number of bins and neurons and making them intergers
num_examples = int(spike_data.shape[0]/example_length)
num_neurons = int(spike_data.shape[1])
#Seperating the data into the defined bins
binslist = []
for y in range(num_examples):
    binslist.append(np.array(neural_data[example_length*(y):example_length*(y+1),:]))
data = np.array(binslist)
#Seperating the test set from
X = data[:,:-1,:]
Y = data[:,-1,:]
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.25,shuffle=False)
test_size = int(len(x_test)/2)
x_validate, y_validate = x_test[:-test_size], y_test[:-test_size]
x_test, y_test = x_test[-test_size:], y_test[-test_size:]
#Compiling the model
model.compile(
    loss=keras.losses.MeanSquaredError(),#reduction="auto", name="mean_squared_error"),
    optimizer=keras.optimizers.Adam(learning_rate=learn_rate), metrics='accuracy')
#Training the model on
history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate), batch_size=batch, epochs=200, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
#Setting predictions and evaluating performance on test set.
y_pred = model.predict(x_test)
results = model.evaluate(x_test, y_test, batch_size=batch)
print("test loss, test acc:", results)

#19 neuron subplots and 1 validation plot
fig = plt.figure(1)
plt.suptitle('Spike Predictions w/ Training and Validation Loss')
a = 4
b = 5
c = 1
d = 1
for i in range(num_neurons):
    neuron = y_test[:,i]
    neuron_pred = y_pred[:,i]
    plt.subplot(a, b, c)
    plt.title('Neuron{}'.format(i))
    plt.plot(neuron,label='True')
    plt.plot(neuron_pred, label='Predicted')
    plt.legend(loc='lower right')
    c = c + 1
plt.subplot(a,b,20)
plt.title('Training and Validation Loss w/Val_Accuracy')
plt.plot(history.history['loss'], color='green', label='Loss')
plt.plot(history.history['val_loss'], color='blue', label='Validation_loss')
# plt.plot(history.history['accuracy'], color='red')
plt.plot(history.history['val_accuracy'], color='red', label='Validation_accuracy')
plt.tight_layout()
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.show()

#scatter plot to show relationship between predicted and true neuron spiking activity.
fig = plt.figure(2)
plt.suptitle('')
for i in range(num_neurons):
    neuron = y_test[:,i]
    neuron_pred = y_pred[:,i]
    neuronx, neuron_predy = np.polyfit(neuron, neuron_pred, 1)
    plt.subplot(a, b, d)
    plt.title('Neuron{}'.format(i))
    plt.scatter(neuron, neuron_pred)
    plt.xlabel('True_spikes', color = 'g')
    plt.ylabel('Predicted_spikes', color = 'y')
    plt.tight_layout()
    plt.plot(neuron, neuronx*neuron+neuron_predy, color='r')
    d = d + 1






