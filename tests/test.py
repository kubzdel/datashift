import unittest

import yaml

from datashift.dataset import Dataset
from tests.test_utils import CleanTextTask, MinTextLengthFilterStrategy, ExampleBalancingTask, MeanValueReduceTask, \
    CountCategoriesTask, CountSubcategoriesPerCateogry


class DatasetIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(input_data_path='./test/sample_data.csv',
                               output_data_dir='/tmp/target',
                               output_file_name='processed',
                               processing_chunksize=5000,
                               input_columns=['full_text'],
                               output_file_size=5000,
                               output_reduce_file_path='./test/reduced.yaml',
                               num_workers=1) \
            .process(CleanTextTask()) \
            .filter(MinTextLengthFilterStrategy(column_name='full_text', min_characters=50)) \
            .balance(ExampleBalancingTask(1, 1)) \
            .reduce(MeanValueReduceTask('mean_words')) \
            .reduce(CountCategoriesTask('count_categories')) \
            .reduce(CountSubcategoriesPerCateogry('count_cat1', 'cat1')) \
            .reduce(CountSubcategoriesPerCateogry('count_cat2', 'cat2')) \
            .reduce(CountSubcategoriesPerCateogry('count_cat3', 'cat3')) \
            .reduce(CountSubcategoriesPerCateogry('count_cat4', 'cat4'))

    def should_create_reduced_file_with_reduced_values(self):
        self.dataset.run()
        reduced_results = self._load_reduced_file()
        self.assertIsNotNone(reduced_results)
        self.assertIn('mean_words', reduced_results)
        self.assertIn('count_categories', reduced_results)
        self.assertIn('count_cat1', reduced_results)
        self.assertIn('count_cat2', reduced_results)
        self.assertIn('count_cat3', reduced_results)
        self.assertIn('count_cat4', reduced_results)
        self.assertAlmostEqual(reduced_results['mean_words'], 5)
        self.assertAlmostEqual(reduced_results['count_categories'], 5)
        self.assertAlmostEqual(reduced_results['count_cat1'], 5)
        self.assertAlmostEqual(reduced_results['count_cat2'], 5)
        self.assertAlmostEqual(reduced_results['count_cat3'], 5)
        self.assertAlmostEqual(reduced_results['count_cat4'], 5)

    def _load_reduced_file(self):
        results_file = open('./test/reduced.yaml')
        return yaml.load(results_file, Loader=yaml.FullLoader)
