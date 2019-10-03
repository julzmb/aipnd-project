import unittest
import unittest.mock
import logging
import train

class MainTests(unittest.TestCase):
    def verify_call(self, funcname):
        with unittest.mock.patch(
            'train.{}'.format(funcname)
        ) as m:
            train.main()
        m.assert_called()
    
    def test_main_calls_getargs(self):
        self.verify_call('getargs')

    def test_main_calls_getdataloaders(self):
        self.verify_call('getdataloaders')

    def test_main_calls_getmodel(self):
        self.verify_call('getmodel')

    def test_main_calls_train(self):
        self.verify_call('train')

    def test_main_calls_test(self):
        self.verify_call('test')

    def test_main_calls_export(self):
        self.verify_call('export')
    

class GetArgsTests(unittest.TestCase):
    @unittest.mock.patch('argparse.ArgumentParser')
    def test_getargs_instantiates_parser(self, mock_class):
        train.getargs()
        mock_class.assert_called()

    @unittest.mock.patch('argparse.ArgumentParser')
    def test_getargs_return_argument_parser(self, mock_class):
        test_parser = unittest.mock.sentinel.fake_parser
        mock_class.return_value = test_parser
        self.assertEqual(
            train.getargs(),
            test_parser
        )


class GetDataLoadersTests(unittest.TestCase):
    pass

class GetModelTests(unittest.TestCase):
    pass

class TrainTests(unittest.TestCase):
    pass

class TestTests(unittest.TestCase):
    pass

class ExportTests(unittest.TestCase):
    pass

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG
    )
    unittest.main()