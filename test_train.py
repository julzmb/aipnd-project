import unittest
import unittest.mock
import logging
import torch
import torchvision
import train

class MainTests(unittest.TestCase):
    def verify_call(self, funcname):
        with unittest.mock.patch(
            'train.{}'.format(funcname)
        ) as m:
            with unittest.mock.patch('torch.utils.data.DataLoader') as _:
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
    def test_getargs_return_argument_parser(self, mock_class):
        test_parser = unittest.mock.MagicMock()
        mock_class.return_value = test_parser
        self.assertEqual(
            train.getargs(),
            test_parser
        )

    def get_args(
        self,
        data_dir='test_data_dir',
        save_dir='test_save_dir',
        arch='vgg16',
        learning_rate='0',
        hidden_units=['0'],
        epochs='0',
        gpu=None
    ):
        args = []
        if data_dir is not None:
            args += [data_dir]

        if save_dir is not None:
            args += ['--save_dir', save_dir]

        if arch is not None:
            args += ['--arch', arch]

        if learning_rate is not None:
            args += ['--learning_rate', learning_rate]

        if hidden_units is not None:
            args += ['--hidden_units']
            args += hidden_units

        if epochs is not None:
            args += ['--epochs', epochs]

        if gpu:
            args+= ['--gpu']

        return train.getargs().parse_args(args)

    def test_getargs_positional_args_datadir(self):
        test_data_dir = 'data/version1/'
        args = self.get_args(
            data_dir=test_data_dir
        )
        self.assertEqual(
            args.data_dir,
            test_data_dir
        )

    def test_getargs_savedir_arg(self):
        test_savedir = 'model_output'
        args = self.get_args(
            save_dir=test_savedir
        )
        self.assertEqual(
            args.save_dir,
            test_savedir
        )

    def test_getargs_savedir_not_included(self):
        with self.assertRaises(SystemExit):
            self.get_args(
                save_dir=None
            )

    def test_getargs_arch_arg(self):
        test_architecture = 'vgg13'
        args = self.get_args(
            arch=test_architecture
        )
        self.assertEqual(
            args.arch,
            test_architecture
        )

    def test_getargs_arch_not_included(self):
        with self.assertRaises(SystemExit):
            self.get_args(
                arch=None
            )

    def test_getargs_arch_not_in_choices(self):
        with self.assertRaises(SystemExit):
            self.get_args(arch='JulianModelArch')

    def test_getargs_learning_rate_arg(self):
        test_learning_rate = '0.0001'
        args = self.get_args(
            learning_rate=test_learning_rate
        )
        self.assertEqual(
            args.learning_rate,
            float(test_learning_rate)
        )

    def test_getargs_learning_rate_not_included(self):
        with self.assertRaises(SystemExit):
            self.get_args(learning_rate=None)

    def run_hidden_units_test(self, test_hidden_units):
        args = self.get_args(
            hidden_units=test_hidden_units
        )
        expected_hidden_units = [int(x) for x in test_hidden_units]
        self.assertEqual(
            args.hidden_units,
            expected_hidden_units
        )

    def test_getargs_hidden_units_arg(self):
        self.run_hidden_units_test(['128'])
    
    def test_getargs_hidden_units_arg_multivalue(self):
        self.run_hidden_units_test(['128', '64'])

    def test_getargs_hidden_units_not_included(self):
        with self.assertRaises(SystemExit):
            self.get_args(hidden_units=None)

    def test_getargs_epochs_arg(self):
        test_epochs = '10'
        args = self.get_args(
            epochs=test_epochs
        )
        self.assertEqual(
            args.epochs,
            int(test_epochs)
        )

    def test_gerargs_epochs_not_included(self):
        with self.assertRaises(SystemExit):
            self.get_args(epochs=None)

    def test_getargs_gpu_arg(self):
        args = self.get_args(
            gpu=True
        )
        self.assertTrue(args.gpu)

    def test_getargs_gpu_not_included(self):
        args = self.get_args()
        self.assertFalse(args.gpu)

class GetDataLoadersTests(unittest.TestCase):
    def test_getdataloaders_retun_type(self):
        train_ret, val_ret, test_ret, = train.getdataloaders('test_datadir', batch_size=1)
        for r in [train_ret, val_ret, test_ret]:
            self.assertEqual(
                type(r),
                torch.utils.data.DataLoader
            )

    @unittest.mock.patch('torch.utils.data.DataLoader')
    def test_getdataloaders_dataloader_constructor_batch_size_arg(self, mock_class):
        batch_sizes = []
        def _dataloader_side_effect(*a, **kw):
            self.assertIn('batch_size', kw)
            batch_sizes.append(
                kw['batch_size']
            )
        mock_class.side_effect = _dataloader_side_effect

        test_batch_size = unittest.mock.sentinel.batch_size
        train.getdataloaders('test_datadir', batch_size=test_batch_size)
        self.assertEqual(
            mock_class.call_count,
            3
        )

        self.assertTrue(
            all([
                x == test_batch_size for x in batch_sizes
            ])
        )

    @unittest.mock.patch('torch.utils.data.DataLoader')
    def test_getdataloaders_dataloader_constructor_dataset_num_workers(self, mock_class):
        num_workers = []
        def _dataloader_side_effect(*a, **kw):
            self.assertIn('num_workers', kw)
            num_workers.append(
                kw['num_workers']
            )
        mock_class.side_effect = _dataloader_side_effect

        train.getdataloaders('test_datadir', batch_size=1)
        self.assertEqual(
            mock_class.call_count,
            3
        )
        self.assertTrue(
            all([
                x == train._DATALOADER_NUM_WORKERS for x in num_workers
            ])
        )

    @unittest.mock.patch('torch.utils.data.DataLoader')
    def test_getdataloaders_dataloader_constructor_dataset_shuffle_arg(self, mock_class):
        shuffles = []
        def _dataloader_side_effect(*a, **kw):
            if 'shuffle' in kw and kw['shuffle']:
                shuffles.append(True)
            else:
                shuffles.append(False)
        mock_class.side_effect = _dataloader_side_effect

        train.getdataloaders('test_datadir', batch_size=1)
        self.assertTrue(any(shuffles))

    @unittest.mock.patch('torch.utils.data.DataLoader')
    def test_getdataloaders_dataloader_constructor_dataset_arg_is_correct_type(self, mock_class):
        def _dataloader_side_effect(*a, **kw):
            dataset, = a
            self.assertEqual(
                type(dataset),
                torchvision.datasets.ImageFolder
            )
        mock_class.side_effect = _dataloader_side_effect
        train.getdataloaders('test_datadir', batch_size=1)




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