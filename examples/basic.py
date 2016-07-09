import sframe as gl
import ffm
from convert import read_libffm_file

# Create SFrames from example text files provided with libffm
trainfile = 'lib/bigdata.tr.txt'
validfile = 'lib/bigdata.te.txt'
train = read_libffm_file(trainfile)
valid = read_libffm_file(validfile)

train['y'] = train['y'].astype(int)
del train['features.0']
valid = valid[train.column_names()]
train.save('examples/small.tr.sframe')
valid.save('examples/small.te.sframe')

# train = gl.SFrame('examples/small.tr.sframe')
# valid = gl.SFrame('examples/small.te.sframe')
print(train)

features = train.column_names()
features.remove('y')
print(features)

# Train a model
m = ffm.FFM(target='y', features=features, eta=0.1, lambda_=0.0)
m.fit(train, valid)

# print(valid)
# yhat = m.predict(valid)
# print(yhat)
