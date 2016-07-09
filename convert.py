import sframe as gl

def read_libffm_file(filename):
    """
    Create an SFrame from a text file in libffm format.

    Parameters
    ----------
    f : str
      Name of the filename.

    Returns
    -------
    out : SFrame
      Each column is of type dict, where keys are integers and values are
      floats.

    Examples
    --------

    >>> train = read_libffm_file('lib/bigdata.tr.txt')
    >>> m = ffm.FFM()
    >>> m.fit(train, target='y')

    """

    def make_dict(z):
        d = {}
        for (f, k, v) in z:
            if f not in d:
                d[f] = {}
            d[f][int(k)] = float(v)
        return d


    x = gl.SFrame.read_csv(filename, header=False)
    x['s'] = x['X1'].apply(lambda x: x.split(' '))
    x['y'] = x['s'].apply(lambda x: int(x[0]))
    x['y'] = x['y'].astype(int)
    x['features'] = x['s'].apply(lambda x: x[1:])
    x['features'] = x['features'].apply(lambda x: [z.split(':') for z in x])
    x['features'] = x['features'].apply(lambda x: make_dict(x))
    sf = x[['y', 'features']]
    return sf.unpack('features')


