import doctest
import unittest
from aidsorb import models


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(models))
    return tests


if __name__ == '__main__':
    unittest.main()
