import unittest
import unittest.mock
import logging
import train

class MainTests(unittest.TestCase):
    def test_is_callable(self):
        self.assertTrue(
            hasattr(train, 'main')
        )
        self.assertTrue(
            callable(train.main)
        )
    
class GetArgsTests(unittest.TestCase):
    def test_is_callable(self):
        self.assertTrue(
            hasattr(train, 'getargs')
        )
        self.assertTrue(
            callable(train.getargs)
        )

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG
    )
    unittest.main()