import doctest
import unittest
from aidsorb import visualize


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(visualize))
    return tests


if __name__ == '__main__':
    unittest.main()
