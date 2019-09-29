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

class GetModelTests(unittest.TestCase):
    def test_getmodel_is_callable(self):
        self.assertTrue(
            hasattr(train, 'getmodel')
        )
        self.assertTrue(
            callable(train.getmodel)
        )

class TrainTests(unittest.TestCase):
    def test_train_is_callable(self):
        self.assertTrue(
            hasattr(train, 'train')
        )
        self.assertTrue(
            callable(train.train)
        )

class ExportTests(unittest.TestCase):
    def test_export_is_callable(self):
        self.assertTrue(
            hasattr(train, 'export')
        )
        self.assertTrue(
            callable(train.export)
        )

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG
    )
    unittest.main()