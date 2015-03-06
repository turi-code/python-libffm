import ffm
from convert import read_libffm_file

# Create SFrames from example text files provided with libffm
trainfile = 'lib/bigdata.tr.txt'
validfile = 'lib/bigdata.te.txt'
train = read_libffm_file(trainfile)
valid = read_libffm_file(validfile)
valid['features.0'] = None
valid = valid[train.column_names()]
train.save('examples/small.tr.sframe')
valid.save('examples/small.te.sframe')

features = [c for c in train.column_names() if c != 'y']

# Train a model
m = ffm.FFM()
m.fit(train, valid, target='y', features=features, nr_iters=15)
yhat = m.predict(valid)
print yhat
