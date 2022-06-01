from dataset import Dataset
from models import LinearRegression, Ridge

_4dm4 = Dataset('4dm4.db')
_4dm4.get_label(['RO32', 'RO16'], 'LN', False).to_csv("test.csv")