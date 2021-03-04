import glob
import os
import unittest

import yaml

from datashift.dataset import Data
from tests.test_utils import CleanTextTask, MinTextLengthFilterStrategy, ExampleBalancingTask, MeanValueReduceTask, \
    CountCategoriesTask, CountSubcategoriesPerCateogry


class DatasetIntegrationTestCase(unittest.TestCase):
    OUTPUT_DIR='./target'
    OUTPUT_METADATA='./target/reduced.yaml'
    INPUT_DATA='./tests/example/*.csv'

    def setUp(self):
        self._cleanup()

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        if os.path.isfile(self.OUTPUT_METADATA):
            os.remove(self.OUTPUT_METADATA)
        files = glob.glob('{}/*.csv'.format(self.OUTPUT_DIR))
        for f in files:
            os.remove(f)

    def test_should_create_reduced_file_with_reduced_values(self):
        Data(input_data_path_pattern=self.INPUT_DATA,
                       output_data_dir_path=self.OUTPUT_DIR,
                       output_file_name_prefix='processed',
                       processing_chunk_size=100,
                       input_columns=["comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult",
                                      "identity_hate"],
                       output_file_size=10,
                       output_metadata_file_path=self.OUTPUT_METADATA,
                       num_workers=1) \
            .process(CleanTextTask()) \
            .filter(MinTextLengthFilterStrategy(column_name='comment_text', min_characters=10)) \
            .balance(ExampleBalancingTask(1, 1)) \
            .reduce(MeanValueReduceTask('mean_words')) \
            .reduce(CountCategoriesTask('count_categories')) \
            .reduce(CountSubcategoriesPerCateogry('count_toxic', 'toxic')) \
            .reduce(CountSubcategoriesPerCateogry('count_severe_toxic', 'severe_toxic')) \
            .reduce(CountSubcategoriesPerCateogry('count_obscene', 'obscene')) \
            .reduce(CountSubcategoriesPerCateogry('count_threat', 'threat')) \
            .reduce(CountSubcategoriesPerCateogry('count_insult', 'insult')) \
            .reduce(CountSubcategoriesPerCateogry('count_insult', 'identity_hate')) \
            .shift()
        reduced_results = self._load_reduced_file()
        self.assertIsNotNone(reduced_results)
        self.assertIn('mean_words', reduced_results)
        self.assertIn('count_categories', reduced_results)
        self.assertIn('count_severe_toxic', reduced_results)
        self.assertIn('count_obscene', reduced_results)
        self.assertIn('count_toxic', reduced_results)
        self.assertIn('count_threat', reduced_results)
        self.assertIn('count_insult', reduced_results)
        self.assertIn('count_insult', reduced_results)

    def test_should_create_correct_number_of_output_files(self):
        Data(input_data_path_pattern=self.INPUT_DATA,
             output_data_dir_path=self.OUTPUT_DIR,
             output_file_name_prefix='processed',
             processing_chunk_size=5,
             input_columns=["comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult",
                            "identity_hate"],
             output_file_size=20,
             num_workers=1) \
            .process(CleanTextTask()) \
            .shift()

        self.assertEqual(len(glob.glob('{}/*.csv'.format(self.OUTPUT_DIR))), 29)


    def test_inference(self):
        data=Data(input_data_path_pattern=self.INPUT_DATA,
             output_data_dir_path=self.OUTPUT_DIR,
             output_file_name_prefix='processed',
             processing_chunk_size=5,
             input_columns=["comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult",
                            "identity_hate"],
             output_file_size=20,
             num_workers=1) \
            .process(CleanTextTask(),inference=True)
        data_sample={'comment_text':'THIS  is... a $Text$.'}

        result=data.inference(data_sample)

        self.assertEqual(result['comment_text'],'this is... a text.')

    def _load_reduced_file(self):
        results_file = open(self.OUTPUT_METADATA)
        return yaml.load(results_file, Loader=yaml.FullLoader)


if __name__ == '__main__':
    unittest.main()