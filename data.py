import os

import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
data_folder_path = os.path.join(dir_path, '..', 'data')
import hashlib
import json
import logging
import pickle
import time

import cx_Oracle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import \
    TextVectorization

AUTOTUNE = tf.data.AUTOTUNE

from collections.abc import Iterable
from pathlib import Path


def create_embedding_matrix(voc, word_index, word_model):
    num_tokens = len(voc) + 2
    hits = 0
    misses = 0
    embedding_dim = len(list(word_model.values())[0])

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = word_model.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix

class DatasetGenerator():

    def __init__(self, train_df, val_df, test_df,
                 numeric_features=[], categorical_features=[], binary_features=[], text_features=[],
                 numeric_labels=[], categorical_labels=[], binary_labels=[],
                 categorical_encoded=False, 
                 word_model=None):
        
        # Define Variables
        #-----------------
        
        # Store the raw data.
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy()

        # Store various nature of features and labels
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.binary_features = binary_features
        self.text_features = text_features
        self.numeric_labels = numeric_labels
        self.categorical_labels = categorical_labels
        self.binary_labels = binary_labels

        self.feature_columns = self.numeric_features + self.categorical_features + self.binary_features + self.text_features
        self.label_columns = self.numeric_labels + self.categorical_labels + self.binary_labels
        
        # Store other parameters
        self.categorical_encoded = categorical_encoded
        self.word_model = word_model
        if word_model:
            self.embedding_dim = len(list(self.word_model.values())[0])
        else:
            self.embedding_dim = 0
        
        # Consolidating the data types of columns to be compatible with tensorflow
        #-------------------------------------------------------------------------
        # Proper data types needed for creating the normalizer and onehotencoder.
        
        self.train_df = self.consolidate_tensorflow_dtypes(self.train_df)
        
        # Get the standardizers for numeric features and labels
        #----------------------------------------------------------
        print(f'----------Adapting numeric features--------------')
        if type(self.numeric_features) is list:
            self.feature_normalizer = preprocessing.Normalization()
            self.feature_normalizer.adapt(self.train_df[self.numeric_features].values)
        elif type(self.numeric_features) is dict:
            self.feature_normalizer = {}
            for key, value in self.numeric_features.items():
                normalizer = preprocessing.Normalization()
                normalizer.adapt(self.train_df[value].values)
                self.feature_normalizer[key] = normalizer

        self.label_normalizer = preprocessing.Normalization()
        self.label_normalizer.adapt(self.train_df[self.numeric_labels].values)
        
        # Consolidating the categorical encoders for categorical features and labels
        #---------------------------------------------------------------------------
        
        print(f'--------Adapting categorical_features------------')
        self.feature_onehotencoders = {}
        self.label_onehotencoders = {}

        for col in self.categorical_features:
            indexer, encoder = self.cat_preprocessing(self.train_df[col])
            self.feature_onehotencoders[col] = {'indexer': indexer, 'encoder': encoder}

        for col in self.categorical_labels:
            indexer, encoder = self.cat_preprocessing(self.train_df[col])
            self.label_onehotencoders[col] = {'indexer': indexer, 'encoder': encoder}
            
        # Build the vectorizers for text features
        #----------------------------------------
        print(f'-------------Adapting text features--------------')
        self.feature_text_vectorizers = {}
        
        for col in self.text_features:
            vectorizer = TextVectorization(
                max_tokens=20000, 
                output_sequence_length=50
            )
            vectorizer.adapt(tf.data.Dataset.from_tensor_slices(self.train_df[[col]].values).batch(128))
            self.feature_text_vectorizers[col] = vectorizer
        
        self.feature_embedding_layers = {}
        for col in self.text_features:
            print(f'Creating embeddings for text feature: {col}')
            voc = self.feature_text_vectorizers[col].get_vocabulary()
            word_index = dict(zip(voc, range(len(voc))))
            embedding_matrix = create_embedding_matrix(voc, word_index, self.word_model)
            embedding_layer = Embedding(
                len(voc) + 2,
                self.embedding_dim,
                embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                trainable=False
            )
            self.feature_embedding_layers[col] = embedding_layer
            
              
        

    def __repr__(self):
        # This is the content when your print a DataGenerator object
        #-----------------------------------------------------------
        rep_str = ''
        
        if self.train_df is not None:
            rep_str += f'train dataset size: {len(self.train_df)} \n'
        if self.val_df is not None:
            rep_str += f'val dataset size: {len(self.val_df)} \n'
        if self.test_df is not None:
            rep_str += f'test dataset size: {len(self.test_df)} \n'
            
        rep_str += '--------------------------------------------------------------------- \n'
        
        if len(self.numeric_features) > 0:
            rep_str += f'All numeric features: {self.numeric_features} \n'
            rep_str += f'Number of numeric features: {len(self.numeric_features)} \n'
        if len(self.categorical_features) > 0:
            rep_str += f'All categorical features: {self.categorical_features} \n'
            rep_str += f'Number of categorical features: {len(self.categorical_features)} \n'
        if len(self.binary_features) > 0:
            rep_str += f'All binary features: {self.binary_features} \n'
            rep_str += f'Number of binary features: {len(self.binary_features)} \n'
        if len(self.numeric_labels) > 0:
            rep_str += f'All numeric labels: {self.numeric_labels} \n'
            rep_str += f'Number of numeric labels: {len(self.numeric_labels)} \n'
        if len(self.categorical_labels) > 0:
            rep_str += f'All categorical labels: {self.categorical_labels} \n'
            rep_str += f'Number of categorical labels: {len(self.categorical_labels)}\n'
        if len(self.binary_labels) > 0:
            rep_str += f'All binary labels: {self.binary_labels} \n'
            rep_str += f'Number of binary labels: {len(self.binary_labels)}'
            
        return rep_str
    
    def consolidate_tensorflow_dtypes(self, df):
        for col in self.numeric_features + self.numeric_labels + self.binary_features + self.binary_labels:
            df[col] = df[col].astype('float32')
            
        for col in self.categorical_features + self.categorical_labels + self.text_features:
            df[col] = df[col].astype(str)
        
        return df
        
    def cat_preprocessing(self, series):
        indexer = preprocessing.StringLookup()
        indexer.adapt(series.values)

        encoder = preprocessing.CategoryEncoding(num_tokens=len(indexer.get_vocabulary()), output_mode='binary')
        encoder.adapt(indexer(series.values))
        print(f'{series.name}: number of categories generated by indexer =  {len(indexer.get_vocabulary())}')

        return indexer, encoder

    def make_dataset(self, df, batch_size=32, shuffle=True):
        input_data, output_data = {}, {}
        
        # Adding dummy label columns if they are not present, this is useful for making a new, unseen dataset
        # This is required if the labels are present in the training dataset, and not present in a new dataset you want to predict on
        for col in self.numeric_labels + self.categorical_labels + self.binary_labels:
            if not col in df:
                df[col] = None
        
        # Make sure the dataframe has dtypes compatible with tensorflow
        df = self.consolidate_tensorflow_dtypes(df)

        if self.numeric_features:
            if type(self.numeric_features) is list:
                input_data['numeric_features'] = df[self.numeric_features].values
            elif type(self.numeric_features) is dict:
                for key, value in self.numeric_features.items():
                    input_data[f'{key}_numeric_features'] = df[value].values

        if self.numeric_labels:
            output_data['numeric_labels'] = self.label_normalizer(df[self.numeric_labels].values)

        if self.binary_features:
            input_data['binary_features'] = df[self.binary_features].values

        if self.binary_labels:
            output_data['binary_labels'] = df[self.binary_labels].values

        for col in self.categorical_features:
            if self.categorical_encoded:
                indexer = self.feature_onehotencoders[col]['indexer']
                encoder = self.feature_onehotencoders[col]['encoder']
                input_data[col] = encoder(indexer(df[col].values))
            else:
                input_data[col] = df[col].values

        for col in self.categorical_labels:
            if self.categorical_encoded:
                indexer = self.label_onehotencoders[col]['indexer']
                encoder = self.label_onehotencoders[col]['encoder']
                output_data[col] = indexer(df[col].values)
            else:
                output_data[col] = df[col].values
        
        for col in self.text_features:
            input_data[col] = df[col].values
        

        # transform them into tensorflow datasets
        input_ds = tf.data.Dataset.from_tensor_slices(input_data)
        if output_data:
            ds = tf.data.Dataset.from_tensor_slices((input_data, output_data))
        else:
            ds = input_ds

        if shuffle:
            ds = ds.shuffle(buffer_size=len(df))

        ds = ds.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

        return ds

    @property
    def train(self, **kwargs):
        return self.make_dataset(self.train_df, **kwargs)

    def get_train_ds(self, **kwargs):
        return self.make_dataset(self.train_df, **kwargs)

    @property
    def val(self):
        return self.make_dataset(self.val_df, shuffle=False)

    def get_val_ds(self, **kwargs):
        return self.make_dataset(self.val_df, shuffle=False, **kwargs)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)

    def get_test_ds(self, **kwargs):
        return self.make_dataset(self.test_df, shuffle=False, **kwargs)

    @property
    def full(self):
        return self.make_dataset(self.train_df.append(self.val_df).append(self.test_df), shuffle=False)

    def get_full_ds(self, **kwargs):
        return self.make_dataset(self.train_df.append(self.val_df).append(self.test_df), shuffle=False, **kwargs)
        

def hash_sha256(df, cols=[]):
    for col in cols:
        df[col] = df[col].apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())
    return df


def get_table_from_oracle(tableName, username=None, password=None, database=None):
    """Establish connection with the oracle database, read a table to a python DataFrame format.
    The credentials are currently sotred as a pickle file in shared drive.

    Args:
        tableName (str): The name of table to be read in oracle
        username (str): credentials to login oracle database (username)
        password (str): credentials to login oracle database (password)
        database (str): name of the database to login

    Return:
        DataFrame: The oracle table that has been read

    """
    if not (username and password and database):
        with open(r'\\tsclient\Z\10_Diversion\wms_credentials.pkl', 'rb') as f:
            username, password, database = pickle.load(f)
    db = cx_Oracle.connect(username, password, database, encoding='UTF-8')
    cursor = db.cursor()
    df = pd.read_sql(f'select * from {tableName}', con=db)
    return df


def bulk_columns_to_type(_df, _source_type=None, _target_type=None):
    if (_source_type is None) or (_target_type is None):
        print('Source type or target type is not specified.')
        return

    _cols = [x for x in _df.columns if _df[x].dtype == _source_type]

    _df[_cols] = _df[_cols].astype(_target_type)

    print(f'A total of {len(_cols)} has been converted from {_source_type} to {_target_type}')

    return _df


def round_columns(_df, cols=[], significance=2, index=False):
    if (cols is None) or (len(cols) == 0):
        print('Please pass in at least one column name / index!')
        print('No changes has been applied to the input DataFrame')
        return _df

    if not isinstance(cols, Iterable):

        print('Please pass columns in an iterable format!')
        print('No changes has been applied to the input DataFrame')
        return _df

    elif (not index) and (not all(isinstance(cols, str))):

        print('When index is sat to False, please pass in a list of columns names as strings')
        print('No changes has been applied to the input DataFrame')
        return _df

    elif index and (not all(isinstance(cols, int))):

        print('When index is sat to True, please pass in a list of columns index as int')
        print('No changes has been applied to the input DataFrame')
        return _df


def consolidate_dtypes(df, string_columns=[], numeric_columns=[], date_columns=[], date_format=''):
    """Format columns in DataFrame into specific data types

    Args:
        df (DataFrame): the DataFrame you wish to format data types
        string_columns (list of str): a list of columns names to be formatted into strings
        numeric_columns (list of str): a list of columns names to be formatted into numbers
        date_columns (list of str): a list of columns names to be formatted into dates
        date_format (str): an optional format string for the date columns. If it is not passed, the date format will be inferred automatically (not advised)

    Returns:
        DataFrame: the preprocessed DataFrame


    """
    logging.info('Consolidating the data types of all columns.')
    #############################
    # Column checking
    #############################

    columns_not_exist = set(string_columns + numeric_columns + date_columns) - set(df.columns)
    logging.info(f'Cannot find columns: {columns_not_exist}')

    columns_not_specified = set(df.columns) - set(string_columns + numeric_columns + date_columns)
    if len(columns_not_specified) != 0:
        logging.warning(f'Data types of these columns have not been specified: {columns_not_specified}')

    #############################
    # Consolidate data types
    #############################

    # string columns
    print(f'Consolidating the data types of string columns: {string_columns}')
    _string_columns = list(set(string_columns) & set(df.columns))
    df[_string_columns] = df[_string_columns].astype('string').replace({'NaN': '', 'nan': '', 'None': ''})
    df[_string_columns] = df[_string_columns].fillna('')

    # numeric columns
    print(f'Consolidating the data types of numeric columns: {numeric_columns}')
    _numeric_columns = list(set(numeric_columns) & set(df.columns))
    df[_numeric_columns] = df[_numeric_columns].apply(pd.to_numeric,
                                                      errors='coerce')  # add error = coerce, invalid parsing will be set as NaN

    # date columns
    print(f'Consolidating the data types of date columns: {date_columns}')
    _date_columns = list(set(date_columns) & set(df.columns))

    if date_format:
        df[_date_columns] = df[_date_columns].apply(pd.to_datetime, format=date_format)
    else:
        df[_date_columns] = df[_date_columns].apply(pd.to_datetime, infer_datetime_format=True)

    print('Data type consolidation completed.')
    return df


def load_pickle(path):
    """ Conveniently load a pickle file

    """
    with open(path, 'rb') as f:
        var = pickle.load(f)

    # If the variable is a DataFrame, display its dimension

    if type(var) == pd.core.frame.DataFrame:
        try:
            print(f"Dimension of the DataFrame: {var.shape}")
        except Exception as e:
            pass

    return var


def load_json(path):
    """Conveniently load a .json file in path.

    Args:
        path (str): The full path of the json file.

    Returns:
        dict: The json file in dictionary format.

    """
    with open(path, encoding='utf-8') as json_file:
        var = json.load(json_file)
    return var


def load_spreadsheet_file(path, string_columns=[], numeric_columns=[], date_columns=[], date_format='',
                          **kwargs):
    """Load data with the correct data types

    Args:
        path (str): the directory of the sbom summary table, can be in csv or excel
        string_columns (list of str): the column names you wnat to treat as string
        numeric_columns (list of str): the column names you want to treat as numbers
        date_columns (list of str): the column names you want to treat as dates
        **kwargs: other parameters to pass to pd.read_excel or pd.read_csv

    Returns:
        DataFrame: the preprocessed DataFrame

    """
    if path:
        path = Path(path)

    file_extension = path.suffix
    assert file_extension.lower() in ['.xlsx', '.xls', '.csv', '.tsv']

    # Get data type converter
    csv_dtypes_converter = {}

    for col in numeric_columns:
        csv_dtypes_converter[col] = lambda x: float(x)
    for col in (string_columns + date_columns):
        csv_dtypes_converter[col] = lambda x: str(x)

    #############################
    # Data loading
    #############################
    print('Reading the file. This might take a while...')

    if file_extension.lower() == '.xlsx' or file_extension.lower() == '.xls':
        # read as excel file
        print('Excel file detected...')
        df = pd.read_excel(path, converters=csv_dtypes_converter, **kwargs)

    elif file_extension.lower() == '.csv':
        # read as csv file
        print('CSV file detected...')
        df = pd.read_csv(path, converters=csv_dtypes_converter, **kwargs)
    elif file_extension.lower() == '.tsv':
        # read as tsv file
        print('TSV file detected...')
        df = pd.read_csv(path, sep='\t', converters=csv_dtypes_converter, **kwargs)

    df = consolidate_dtypes(df, string_columns, numeric_columns, date_columns, date_format=date_format)

    print(f'Dimension of the input file: #Rows={df.shape[0]}, #Columns={df.shape[1]}')

    return df


def archive_and_mkdir(path):
    """Create a folder. If the file already exist, rename the old file
       with an extension of the last modified date, and create the folder
       with the desired name.
    
    Args:
        path (str): The path of the folder to make.
    
    Returns:
        None

    """
    if os.path.exists(path):
        print(f'Folder already exists: {path}')
        existing_path = path
        existing_folder_modify_time = time.strftime('%Y%m%d%H%M%S', time.localtime(os.path.getmtime(existing_path)))
        archiving_path = f'{existing_path}_{existing_folder_modify_time}'
        os.rename(existing_path, archiving_path)
        print(f'Renaming to: {archiving_path}')

    os.mkdir(path)
    print(f'mkdir: {path} successful')


def archive_and_save(var, path):
    """Save pickle file. If the file already exist, rename the old file 
        with an extension of the last modified date, and save the pickle file
        with the desired name.
        
    Args:
        var (obj): any python object you want to save
        
    Returns:
        None
    
    """
    if os.path.exists(path):
        print(f'File already exists: {path}')
        existing_path = path
        existing_filename, existing_file_extension = os.path.splitext(existing_path)
        existing_file_modify_time = time.strftime('%Y%m%d%H%M%S', time.localtime(os.path.getmtime(existing_path)))
        archiving_path = f'{existing_filename}_{existing_file_modify_time}{existing_file_extension}'
        os.rename(existing_path, archiving_path)
        print(f'Renaming to: {archiving_path}')

    if type(var) == pd.core.frame.DataFrame:
        try:
            print(f"Dimension of the DataFrame: {var.shape}")
        except Exception as e:
            pass

    with open(path, 'wb') as f:
        pickle.dump(var, f)


def data_ingestion(path, string_columns=[], numeric_columns=[], date_columns=[], date_format=''):
    '''Full data ingestion pipeline

    '''
    # Read the file with confirmed datatypes for each column
    filename, file_extension = os.path.splitext(path)
    if file_extension.lower() in ['.xlsx', '.xls', '.csv', '.tsv']:
        df = load_spreadsheet_file(path, file_extension, string_columns, numeric_columns, date_columns, date_format)
    elif file_extension.lower() == '.pkl':
        df = load_pickle(path)

    # Save a copy of the file in the data folder in pickle format
    backup_dir = os.path.join(data_folder_path, os.path.basename(filename) + '.pkl')
    archive_and_save(df, backup_dir)

    return df


if __name__ == '__main__':
    pass
