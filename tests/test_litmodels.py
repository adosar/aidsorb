import doctest
import unittest
from aidsorb import litmodels


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(litmodels))
    return tests


if __name__ == '__main__':
    unittest.main()
