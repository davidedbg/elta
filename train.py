import pandas as pd
import opendatasets as od
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import FunctionTransformer
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential


NUM_EPOCHS = 30
BATCH_SIZE = 32
CRITERION = nn.BCELoss()
MODEL_PATH = "model_weights.pth"
DS_URL = "https://www.kaggle.com/competitions/titanic/data"
DS_DIR = 'data/train.csv'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(0)


def download_dataset(ds_url, current_directory):
    try:
        od.download(ds_url, data_dir=current_directory)
    except Exception as e:
        print(f'Error: {e}')


def preprocess(dataset):
    """
    preprocess the data
    :param dataset: dataframe of dataset
    :return: dataset after preprocessing
    """
    # drop columns with no actual meaning to prediction
    dataset = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # deal numerical and categorical columns separately
    numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    categoricl_cols = ['Pclass', 'Sex', 'Embarked']
    dataset = dataset[numerical_cols + categoricl_cols]

    # replace missing numerical values with median and use minmax scaler
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=pd.NA, strategy='median')),
        ('min_max', MinMaxScaler(feature_range=(0, 1)))])

    # replace missing categorical values with the most frequent value and convert all to one hot vector
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=pd.NA, strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categoricl_cols)])
    preprocessor.fit(dataset)
    dataset = pd.DataFrame(preprocessor.transform(dataset))
    onehot_col_names = list(preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categoricl_cols))
    preproc_col_names = numerical_cols + onehot_col_names
    dataset.columns = preproc_col_names
    return dataset


def show_EDA(dataset):
    dataset.head(5)
    missing_values_count = dataset.isnull().sum()
    print(f'shape: {dataset.shape}')
    print(f'number of missing values:\n{missing_values_count}')
    unique_count = dataset.nunique()
    print(f'number of unique values:\n{unique_count}')
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    sns.countplot(data=dataset, x='Survived', ax=axes[0, 0])
    sns.countplot(data=dataset, x='Pclass', ax=axes[0, 1])
    sns.countplot(data=dataset, x='Sex', ax=axes[0, 2])
    sns.countplot(data=dataset, x='SibSp', ax=axes[0, 3])
    sns.countplot(data=dataset, x='Parch', ax=axes[1, 0])
    sns.countplot(data=dataset, x='Embarked', ax=axes[1, 1])
    sns.histplot(dataset['Fare'], kde=True, ax=axes[1, 2])
    sns.histplot(dataset['Age'].dropna(), kde=True, ax=axes[1, 3])
    plt.show()
    figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))
    dataset.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axesbi[0, 0], xlim=[0, 1])
    dataset.groupby('SibSp')['Survived'].mean().plot(kind='bar', ax=axesbi[0, 1], xlim=[0, 1])
    dataset.groupby('Parch')['Survived'].mean().plot(kind='bar', ax=axesbi[0, 2], xlim=[0, 1])
    dataset.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axesbi[0, 3], xlim=[0, 1])
    dataset.groupby('Embarked')['Survived'].mean().plot(kind='bar', ax=axesbi[1, 0], xlim=[0, 1])
    sns.boxplot(x="Survived", y="Age", data=dataset, ax=axesbi[1, 1])
    # use log scale in fare to see the difference more clear (outlier stretches the range)
    sns.boxplot(x="Survived", y="Fare", data=dataset, ax=axesbi[1, 2], log_scale=True)
    plt.show()


class TabularDataset(Dataset):
    """
    data generator
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


class Model(torch.nn.Module):
    """
    model class. batchnorm and dropouts appeared to make no improvement
    """
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(12, 8)
        self.linear2 = torch.nn.Linear(8, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.dropout = torch.nn.Dropout()
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)
        # x = self.dropout(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


def arange_data(dataset):
    """
    :param dataset: dataset generator
    :return: train, validation and test DataLoader objects
    """
    X = dataset.drop(['Survived'], axis=1)
    columns = X.columns
    y = dataset['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, stratify=y, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=2)

    # save train, val and test separately for use in the streamlit app
    save_splitted_data(X_train, y_train, columns, 'train_only.csv')
    save_splitted_data(X_val, y_val, columns, 'validation_only.csv')
    save_splitted_data(X_test, y_test, columns, 'test_only.csv')

    X_train = preprocess(pd.DataFrame(X_train, columns=columns))
    X_val = preprocess(pd.DataFrame(X_val, columns=columns))
    X_test = preprocess(pd.DataFrame(X_test, columns=columns))
    X_train = X_train[:].values
    X_val = X_val[:].values
    X_test = X_test[:].values
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test, y_test)

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=X_val.shape[0],
                        shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=X_test.shape[0],
                         shuffle=False)
    return train_dl, val_dl, test_dl


def save_splitted_data(X, y, columns, path):
    new_data = pd.DataFrame(X)
    new_data.columns = columns
    new_data['Survived'] = pd.DataFrame(y)
    new_data.to_csv(path)



def train_model(model, train_dl, val_dl):
    """
    :param model: pytorch model
    :param train_dl: dataloader for training
    :param val_dl: dataloader for validation
    :return: train loss, validation loss and validation accuracy
    """
    train_loss = []
    val_loss = []
    val_accuracy = []
    for epoch in range(NUM_EPOCHS):
        train_loss_batch = []
        val_loss_batch = []
        for i, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs.float())
            loss = CRITERION(output, targets.unsqueeze(1).float())

            ######### optional change #############
            # adding the following line penalize uncertainty and values close to zero.
            # it gives lower overall accuracy but better balance between "0" label and "1" label
            # as can be seen in the prediction distribution and confusion matrix made by streamlit

            # loss = loss + (1-abs(output - 0.5) + (1- output)).mean()

            train_loss_batch.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        train_loss.append(sum(train_loss_batch) / i)
        for (inputs, targets) in val_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs.float())
            loss = CRITERION(output, targets.unsqueeze(1).float())
            val_loss.append(loss.cpu().detach().numpy())
            actual = targets.cpu().numpy()
            actual = actual.reshape((len(actual), 1))
            output = output.cpu().detach().numpy()
            output = np.where(output >= 0.5, 1, 0)
            val_accuracy.append(accuracy_score(actual, output))
    print("################### Training finished ##########################")
    return train_loss, val_loss, val_accuracy


def train_val_stats(train_loss, val_loss, val_accuracy):
    """
    shows graphs of train and validation loss and validation accuracy
    :param train_loss: train loss (returned by train_model method)
    :param val_loss: validation loss (returned by train_model method)
    :param val_accuracy: validation accuracy (returned by train_model method)
    """
    x = list(range(1, NUM_EPOCHS + 1))
    plt.figure()
    plt.plot(np.array(x), np.array(train_loss), label='train_loss')
    plt.plot(np.array(x), np.array(val_loss), label='eval_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    x = list(range(1, NUM_EPOCHS + 1))
    plt.figure()
    plt.plot(np.array(x), np.array(val_accuracy), label='validation set accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def compute_accuracy(model, data_loader, train_or_test: str):
    """
    computes the accuracy of the model
    :param data_loader: DataLoader object of the data
    :param train_or_test: a string ("train" or "test" spcifying wether to print accuracy for train or test
    """
    predictions, actuals = [], []
    for (inputs, targets) in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        yhat = model(inputs.float())
        yhat = yhat.cpu().detach().numpy()
        actual = targets.cpu().numpy()
        actual = actual.reshape((len(actual), 1))
        yhat = np.where(yhat >= 0.5, 1, 0)
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    print(train_or_test + ': ', acc)


if __name__ == '__main__':
    # download_dataset(ds_url, DS_DIR)
    try:
        train = pd.read_csv(DS_DIR)
    except Exception as e:
        print(f'Error: {e}')
    # show_EDA(train)
    train_dl, val_dl, test_dl = arange_data(train)

    model = Model()
    model.to(device)
    optimizer = optim.Adam(model.parameters())

    train_loss, val_loss, val_accuracy = train_model(model, train_dl, val_dl)

    # Save the model's state_dict
    try:
        torch.save(model.state_dict(), MODEL_PATH)
    except Exception as e:
        print(f'Error: {e}')
    # load model from disc
    loaded_model = Model()
    loaded_model.to(device)
    try:
        loaded_model.load_state_dict(torch.load(MODEL_PATH))
    except Exception as e:
        print(f'Error: {e}')

    train_val_stats(train_loss, val_loss, val_accuracy)

    compute_accuracy(loaded_model, train_dl, 'train')
    compute_accuracy(loaded_model, train_dl, 'test')





