import doctest
import unittest
from aidsorb import transforms


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(transforms))
    return tests


if __name__ == '__main__':
    unittest.main()
