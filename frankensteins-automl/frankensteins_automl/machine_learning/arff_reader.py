import arff
import numpy
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def read_arff(path, target_column_index):
    data = numpy.array(arff.load(open(path, "r"))["data"])
    data_x = data[:, :target_column_index]
    data_encoder = OrdinalEncoder()
    data_x = data_encoder.fit_transform(data_x)
    data_x = data_x.astype(numpy.float64)
    data_y = data[:, target_column_index]
    class_encoder = LabelEncoder()
    data_y = class_encoder.fit_transform(data_y)
    data_y = data_y.astype(numpy.float64)
    return data_x, data_y, data_encoder, class_encoder
