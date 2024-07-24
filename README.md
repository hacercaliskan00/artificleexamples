# Water Quality Prediction


This project aims to predict water quality based on various parameters using a neural network model implemented with TensorFlow and Keras. The dataset used in this project contains water quality metrics and corresponding zones, and the model aims to classify the water quality zone.


# Table of Contents
Dataset
Preprocessing
Model
Training
Evaluation
Dependencies
Usage
Contributing
License


# Dataset
The dataset used in this project includes various features related to water quality. The features include measurements like pH, turbidity, hardness, etc. The target variable is the zone which indicates the water quality zone.

# Preprocessing
Loading the Data:

  
    import pandas as pd
    df = pd.read_csv('water.csv')

    

# Handling Missing Values:

    print(df.isnull().sum())
    
    
# Feature Selection:
    X = df.drop(columns=['Sta_ID', 'Date', 'Zone'])
    y = df['Zone']



# Encoding Categorical Variables:

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X = pd.get_dummies(X)


# Normalization:

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


# Splitting the Data:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# Model

The neural network model is built using TensorFlow and Keras. It consists of three dense layers:

Input layer with 64 units and ReLU activation.
Hidden layer with 32 units and ReLU activation.
Output layer with 1 unit and sigmoid activation for binary classification.


    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])






# Training
The model is trained with early stopping to prevent overfitting. The training process uses an 80-20 split for validation.

    from tensorflow.keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping], batch_size=32)


# Evaluation
Evaluate the model performance on the test set:

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")



# Dependencies

pandas
numpy
scikit-learn
tensorflow
Install the required packages using pip:

    pip install pandas numpy scikit-learn tensorflow


# Usage
Clone the repository:

    git clone https://github.com/hacercaliskan/artificleexampless.git


# Navigate to the project directory:

      cd artificleexampless.ipynb


# License
This project is licensed under the MIT License. See the LICENSE file for details.




















