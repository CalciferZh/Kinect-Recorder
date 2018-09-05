import pickle


def pickle_load(path):
  """
  Wrapper to load from pickle file.

  Parameter
  ---------
  path: Path of the pickle file to be loaded.

  """
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data


def pickle_save(path, data):
  """
  Save data into pickle file.

  Parameter
  ---------
  path: Path of the pickle file to save.

  data: Data to be saved.

  """
  with open(path, 'wb') as f:
    pickle.dump(data, f)
