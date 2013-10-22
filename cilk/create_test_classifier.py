import numpy as np
import imread
import pylab

print dir(imread)

classifier = np.vstack([np.arange(256) / 256.0 * i for i in range(16)])

classifier = classifier.astype(np.uint16)
imread.imsave('test_classifier.tif', classifier)

pylab.imshow(classifier)
pylab.colorbar()
pylab.show()
