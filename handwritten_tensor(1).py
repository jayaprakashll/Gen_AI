from sklearn.base import BaseEstimator
from keras.src.models import Sequential
from keras.src.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.src.datasets import mnist
from keras.src.utils import to_categorical
from keras import optimizers

def create_cnn_model(dropout_rate=0.5):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_mlp_model(dropout_rate=0.5):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class KerasClassifierWrapper(BaseEstimator):
    def __init__(self, build_fn, epochs=10, batch_size=8, dropout_rate=0.5, **kwargs):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs

    def fit(self, X, y, validation_data=None, **kwargs):
        self.model = self.build_fn(dropout_rate=self.dropout_rate, **self.kwargs)
        if validation_data:
            val_data = validation_data
        else:
            val_data = None
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_data=val_data, **kwargs)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

cnn_model = KerasClassifierWrapper(build_fn=create_cnn_model, epochs=10, batch_size=128, verbose=0)
param_grid_cnn = {'dropout_rate': [0.3, 0.4, 0.5]}
grid_cnn = GridSearchCV(estimator=cnn_model, param_grid=param_grid_cnn, n_jobs=1, cv=3)
grid_cnn_result = grid_cnn.fit(X_train, y_train, validation_data=(X_test, y_test),
                               callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

print("Best CNN Params: ", grid_cnn_result.best_params_)
print("Best CNN Cross-Validation Accuracy: ", grid_cnn_result.best_score_)

best_cnn_model = grid_cnn_result.best_estimator_.model
cnn_test_loss, cnn_test_acc = best_cnn_model.evaluate(X_test, y_test, verbose=0)
print("CNN Test Accuracy: ", cnn_test_acc)

mlp_model = KerasClassifierWrapper(build_fn=create_mlp_model, epochs=10, batch_size=128, verbose=0)
param_grid_mlp = {'dropout_rate': [0.3, 0.4, 0.5]}
grid_mlp = GridSearchCV(estimator=mlp_model, param_grid=param_grid_mlp, n_jobs=-1, cv=3)
grid_mlp_result = grid_mlp.fit(X_train, y_train, validation_data=(X_test, y_test),
                               callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

print("Best MLP Params: ", grid_mlp_result.best_params_)
print("Best MLP Cross-Validation Accuracy: ", grid_mlp_result.best_score_)

best_mlp_model = grid_mlp_result.best_estimator_.model
mlp_test_loss, mlp_test_acc = best_mlp_model.evaluate(X_test, y_test, verbose=0)
print("MLP Test Accuracy: ", mlp_test_acc)
