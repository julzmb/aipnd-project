import unittest
import unittest.mock
import logging
import train

class MainTests(unittest.TestCase):
    def test_main_is_callable(self):
        self.assertTrue(
            hasattr(train, 'main')
        )
        self.assertTrue(
            callable(train.main)
        )
    
    def test_main_calls_getargs(self):
        with unittest.mock.patch('train.getargs') as m:
            train.main()
        m.assert_called()

    def test_main_calls_getmodel(self):
        with unittest.mock.patch('train.getmodel') as m:
            train.main()
        m.assert_called()

    def test_main_calls_train(self):
        with unittest.mock.patch('train.train') as m:
            train.main()
        m.assert_called()

    def test_main_calls_export(self):
        with unittest.mock.patch('train.export') as m:
            train.main()
        m.assert_called()
    
class GetArgsTests(unittest.TestCase):
    def test_getargs_is_callable(self):
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