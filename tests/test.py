import glob
import os
import unittest

import yaml

from datashift.datapipeline import DefaultCSVSaver, DefaultCSVReader, DataPipeline, DefaultTextLineReader, \
    DefaultTextLineSaver
from tests.test_utils import CleanTextTask, MinTextLengthFilterStrategy, ExampleBalancingTask, MeanValueReduceTask, \
    CountCategoriesTask, CountSubcategoriesPerCateogry, CleanTextLineTask


class DatasetIntegrationTestCase(unittest.TestCase):
    OUTPUT_DIR = './target'
    OUTPUT_METADATA = './target/reduced.yaml'
    INPUT_DATA_CSV = './tests/example/*.csv'
    INPUT_DATA_TXT = './tests/example/*.txt'

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
        DataPipeline(reader=DefaultCSVReader(input_data_path_pattern=self.INPUT_DATA_CSV,
                                             input_columns=["comment_text", "toxic", "severe_toxic", "obscene",
                                                            "threat", "insult",
                                                            "identity_hate"]),
                     saver=DefaultCSVSaver(output_data_dir_path=self.OUTPUT_DIR, output_file_size=10,
                                           output_file_name_prefix='processed'),
                     processing_chunk_size=100,
                     output_metadata_file_path=self.OUTPUT_METADATA,
                     num_workers=1) \
            .process_task(CleanTextTask()) \
            .filter_task(MinTextLengthFilterStrategy(column_name='comment_text', min_characters=10)) \
            .balance_task(ExampleBalancingTask(1, 1)) \
            .reduce_task(MeanValueReduceTask('mean_words')) \
            .reduce_task(CountCategoriesTask('count_categories')) \
            .reduce_task(CountSubcategoriesPerCateogry('count_toxic', 'toxic')) \
            .reduce_task(CountSubcategoriesPerCateogry('count_severe_toxic', 'severe_toxic')) \
            .reduce_task(CountSubcategoriesPerCateogry('count_obscene', 'obscene')) \
            .reduce_task(CountSubcategoriesPerCateogry('count_threat', 'threat')) \
            .reduce_task(CountSubcategoriesPerCateogry('count_insult', 'insult')) \
            .reduce_task(CountSubcategoriesPerCateogry('count_insult', 'identity_hate')) \
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
        DataPipeline(reader=DefaultCSVReader(input_data_path_pattern=self.INPUT_DATA_CSV,
                                             input_columns=["comment_text", "toxic", "severe_toxic", "obscene",
                                                            "threat", "insult",
                                                            "identity_hate"]),
                     saver=DefaultCSVSaver(output_data_dir_path=self.OUTPUT_DIR, output_file_size=20,
                                           output_file_name_prefix='processed'),
                     processing_chunk_size=5,
                     num_workers=1) \
            .process_task(CleanTextTask()) \
            .shift()

        self.assertEqual(len(glob.glob('{}/*.csv'.format(self.OUTPUT_DIR))), 15)

    def test_inference(self):
        data = DataPipeline(reader=DefaultCSVReader(input_data_path_pattern=self.INPUT_DATA_CSV,
                                                    input_columns=["comment_text", "toxic", "severe_toxic", "obscene",
                                                                   "threat", "insult",
                                                                   "identity_hate"]),
                            saver=DefaultCSVSaver(output_data_dir_path=self.OUTPUT_DIR, output_file_size=20,
                                                  output_file_name_prefix='processed'),
                            processing_chunk_size=5, \
                            num_workers=1) \
            .process_task(CleanTextTask(), inference=True)
        data_sample = {'comment_text': 'THIS  is... a $Text$.'}

        result = data.inference(data_sample)

        self.assertEqual(result['comment_text'], 'this is... a text.')

    def _load_reduced_file(self):
        results_file = open(self.OUTPUT_METADATA)
        return yaml.load(results_file, Loader=yaml.FullLoader)


if __name__ == '__main__':
    unittest.main()
