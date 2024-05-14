import doctest
import unittest
from aidsorb import visualize


class TestDrawPCDFromFile(unittest.TestCase):
    def test_draw(self):
        visualize.draw_pcd_from_file('tests/samples/IRMOF-1.xyz')


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(visualize))
    return tests


if __name__ == '__main__':
    unittest.main()
