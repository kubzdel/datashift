from __future__ import annotations

import csv
import glob
import json
import logging
import math
import multiprocessing
import os
import pickle
import sys
import tempfile
import time
import traceback
from abc import abstractmethod, ABC
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import yaml

from datashift.task import AbstractBalancingTask, TaskType, AbstractProcessingTask, AbstractFilterTask


class AbstractDataBucket(ABC):
    def __init__(self, data_buckets):
        self.data_buckets = data_buckets

    @abstractmethod
    def next_data_chunk(self):
        raise NotImplementedError("Method not implemented!")

    def setup(self):
        pass

    def teardown(self):
        pass

    def __str__(self):
        return str(self.data_buckets)


class AbstractReader(ABC):
    @abstractmethod
    def build_data_buckets(self, pool, chunksize):
        raise NotImplementedError("Method not implemented!")


class AbstractFileDataBucketLoader(AbstractDataBucket):
    @abstractmethod
    def _read_data_bucket(self, filepath, from_item, chunk_size):
        raise NotImplementedError("Method not implemented!")

    def next_data_chunk(self):
        data_list = []
        if len(self.data_buckets) > 0:
            assert type(self.data_buckets) == list
            for data_bucket in self.data_buckets:
                (filepath, from_item, chunk_size) = data_bucket
                data_chunk = self._read_data_bucket(filepath, from_item, chunk_size)
                assert type(data_chunk) == list
                data_list.append(data_chunk)
                self.data_buckets.remove(data_bucket)
        return [item for sublist in data_list for item in sublist]


class AbstractFileReader(AbstractReader):
    def __init__(self, input_data_path_pattern):
        self.input_data_path_pattern = input_data_path_pattern

    @abstractmethod
    def _determine_number_of_items_in_file(self, path, chunk_size) -> int:
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def _create_data_bucket_object(self, data_buckets):
        raise NotImplementedError("Method not implemented!")

    def _no_items_in_one_file(self, path_and_chunk_size):
        path, chunk_size = path_and_chunk_size
        return self._determine_number_of_items_in_file(path, chunk_size)

    def build_data_buckets(self, pool, chunk_size) -> list:
        all_file_paths = glob.glob(self.input_data_path_pattern)
        number_of_items_per_file = pool.map(self._no_items_in_one_file, [(path, chunk_size) for path in all_file_paths])
        data_buckets = []
        remaining_to_full_chunk = chunk_size
        for file, no_elements in zip(all_file_paths, number_of_items_per_file):
            buffer_level = 0
            if remaining_to_full_chunk < chunk_size:
                current_chunk_size = min(remaining_to_full_chunk, no_elements)
                data_buckets[-1].append((file, buffer_level, current_chunk_size))
                buffer_level += current_chunk_size
                remaining_to_full_chunk -= current_chunk_size

            while buffer_level != no_elements:
                remaining_to_full_chunk = chunk_size
                if buffer_level + chunk_size <= no_elements:
                    data_buckets.append([(file, buffer_level, chunk_size)])
                    buffer_level += chunk_size
                    remaining_to_full_chunk -= chunk_size
                else:
                    data_buckets.append([(file, buffer_level, no_elements - buffer_level)])
                    remaining_to_full_chunk -= (no_elements - buffer_level)
                    buffer_level += (no_elements - buffer_level)
        return [self._create_data_bucket_object(db) for db in data_buckets]


class DefaultCSVDataBucketLoader(AbstractFileDataBucketLoader):
    def __init__(self, data_buckets, input_columns):
        self.input_columns = input_columns
        super().__init__(data_buckets=data_buckets)

    def _read_data_bucket(self, filepath, from_item, chunk_size):
        df = pd.read_csv(filepath, skiprows=range(1, max(from_item, 1)), nrows=chunk_size,
                         usecols=self.input_columns).fillna('')
        return df.to_dict('records')


class DefaultCSVReader(AbstractFileReader):
    def __init__(self, input_data_path_pattern, input_columns):
        self.input_columns = input_columns
        super().__init__(input_data_path_pattern)

    def _determine_number_of_items_in_file(self, path, chunk_size) -> int:
        with open(path, 'r', encoding='utf-8') as f:
            first_key = next(csv.reader(f))[0]
        df_iter = pd.read_csv(path, usecols=[first_key], chunksize=chunk_size)
        for i, df in enumerate(df_iter):
            continue
        return chunk_size * i + df.shape[0]

    def _create_data_bucket_object(self, data_buckets):
        return DefaultCSVDataBucketLoader(data_buckets=data_buckets, input_columns=self.input_columns)


class DefaultTextLineBucketLoader(AbstractFileDataBucketLoader):
    def _read_data_bucket(self, filepath, from_item, chunk_size):
        data_chunk = []
        to_item = from_item + chunk_size
        fp = open(filepath)
        for i, line in enumerate(fp):
            if from_item <= i < to_item:
                data_chunk.append(line)
            elif i == to_item:
                break
        fp.close()
        return data_chunk


class DefaultTextLineReader(AbstractFileReader):
    def __init__(self, input_data_path_pattern):
        super().__init__(input_data_path_pattern)

    def _determine_number_of_items_in_file(self, path, chunk_size) -> int:
        counter = 0
        for _ in open(path):
            counter += 1
        return counter

    def _create_data_bucket_object(self, data_buckets):
        return DefaultTextLineBucketLoader(data_buckets)


class AbstractSaver(ABC):
    @abstractmethod
    def save(self, data):
        raise NotImplementedError("Method not implemented!")


class AbstractFileSaver(AbstractSaver):
    def __init__(self, output_data_dir_path, output_file_name_prefix, output_file_size,
                 saving_status_file_prefix='.saving_status'):
        self.output_data_dir_path = output_data_dir_path
        self.output_file_name_prefix = output_file_name_prefix
        self.output_file_size = output_file_size
        self.saving_status_file_prefix = saving_status_file_prefix
        if self.saving_status_file_prefix in self.output_file_name_prefix:
            raise Exception('The saving status file prefix ({}) can be a sub-name of output file name ({})',
                            self.saving_status_file_prefix, self.output_file_name_prefix)
        self.logger = logging.getLogger('datashift')

    @abstractmethod
    def _save_chunk(self, data, filename, new_file):
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def _file_extension(self):
        raise NotImplementedError("Method not implemented!")

    def _generate_file_path(self, pid, file_nr) -> str:
        return os.path.join(self.output_data_dir_path,
                            '{}_{}_{}.{}'.format(self.output_file_name_prefix, pid, file_nr, self._file_extension()))

    def save(self, data) -> None:
        if self.output_data_dir_path and not os.path.exists(self.output_data_dir_path):
            os.makedirs(self.output_data_dir_path)
        pid = os.getpid()
        saving_status_file_path = '{}/{}_{}'.format(self.output_data_dir_path, self.saving_status_file_prefix, pid)
        last_file_path = None
        remaining = 0
        if os.path.isfile(saving_status_file_path):
            with open(saving_status_file_path, 'r') as f:
                last_file_path, last_file_items_str = f.readline().split(';')
                last_file_items = int(last_file_items_str)
                remaining = self.output_file_size - last_file_items
        if last_file_path and remaining > 0:
            self._save_chunk(data[:remaining], last_file_path, False)
            last_file_items += len(data[:remaining])

        if len(data) > remaining:
            for data_part in self._chunk_by_n_rows(data[remaining:], self.output_file_size):
                next_file_nr = len(glob.glob(self._generate_file_path(pid, '*'))) + 1
                file_path = self._generate_file_path(pid, next_file_nr)
                self._save_chunk(data_part, file_path, True)
                last_file_path = file_path
                last_file_items = len(data_part)
        with open(saving_status_file_path, 'w') as f:
            f.writelines(['{};{}'.format(last_file_path, last_file_items)])

    def _chunk_by_n_rows(self, data_list, size) -> tuple:
        return (data_list[pos:pos + size] for pos in range(0, len(data_list), size))

    def clean_savings_statuses(self) -> None:
        saving_status_generic_file_path = '{}/{}_{}'.format(self.output_data_dir_path, self.saving_status_file_prefix,
                                                            '*')
        for file_path in glob.glob(saving_status_generic_file_path):
            os.remove(file_path)


class DefaultCSVSaver(AbstractFileSaver):
    def __init__(self, output_data_dir_path, output_file_name_prefix, output_file_size):
        super().__init__(output_data_dir_path, output_file_name_prefix, output_file_size)

    def _file_extension(self):
        return 'csv'

    def _save_chunk(self, data, filepath, new_file):
        keys = data[0].keys()
        with open(filepath, mode='a', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, keys, quoting=csv.QUOTE_ALL)
            if new_file:
                dict_writer.writeheader()
            dict_writer.writerows(data)


class DefaultTextLineSaver(AbstractFileSaver):
    def __init__(self, output_data_dir_path, output_file_name_prefix, output_file_size):
        super().__init__(output_data_dir_path, output_file_name_prefix, output_file_size)

    def _file_extension(self):
        return 'txt'

    def _save_chunk(self, data, filepath, new_file):
        with open(filepath, mode='a', encoding='utf-8') as output_file:
            for line in data:
                output_file.write("{}{}".format(line, os.linesep))


class DataPipeline:
    """
        Lightweight and generic data processor that allows quickly filtering, balancing and processing a data set from one form to another.
        Especially useful for processing data from its original form to a form ready for advanced analysis and machine learning.
        It is possible to automatically generate final dataset statistics (metadata) after processing, i.e. number of observations in each class or in the whole dataset,
        which can be useful when adjusting a learning rate optimization during training of a machine learning model.
        List of features:
        -   loading data from multiple csv files,
        -   multi-thread processing of records,
        -   filtering out unwanted records,
        -   saving processed data to multiple files,
        -   balancing the proportion of observations in each class not only by number of elements but also by characteristics of the elements in each class (e.g. same proportion of color and grayscale images in each class),
        -   computing (reducing) numeric values - metadata of the dataset (e.g. number of elements in each class or minimum / maximum number of words in sentences),
        -   saving reduced values to yaml / json files.

        Args:
            input_data_path_pattern (str): Path to the folder where raw data are located
            output_data_dir_path (str): Path to the folder where finall processed data should be saved
            output_file_name_prefix (str): Generic filename prefix of all output (final) files
            processing_chunk_size (int): Number of observation that are processed at the same time. Higher chunk size use more RAM. (Default 20000).
            input_columns (lint[str]): List of column names to be read from the raw input files. To minimize memory consumption in need to load only required columns.
            output_file_size (int): Total number of records / observations written to a single output file. If the number of processed samples exceeds this value, a next new data file is created. (Default: 50000).
            num_workers (int): Number of processing threads. Higher number of threads results in faster processing and higher RAM consumption.
            output_reduce_file_path (str): Path to the file with reduced values (metadata). The file is only created when reducing tasks have been added for processing.
            output_reduce_file_format (str): Format of the file with reduced values (metadata). The file is only created when reducing tasks have been added for processing.
            saving_status_file_prefix (str): Location of a temporary file containing information about the saving status of processed results. The file is deleted after processing is complete and the output dataset is created. (Default: '.saving_status').
            verbose (bool): If it is set to True, processing statusses are logged to the logger during processing. If false, the logger is not used. (Default: True).
            logger: Custom logger to which logs are passed on. If the logger is undefined and verbose = True then a default logger is created to print statuses on the console. (Default: None).
        """

    _FLATTEN_ORDER_EXCEPTION = "The flatten task can only be executed after a processing stage. Currently the are no processing tasks assigned"

    def __init__(self, reader, saver=None, processing_chunk_size=20000, num_workers=None,
                 output_reduce_file_path=None, output_reduce_file_format='yaml', custom_reduce_save_callback=None,
                 verbose=True, logger=None):
        self.flattened_steps = []
        assert output_reduce_file_format == 'yaml' or output_reduce_file_format == 'json' 'Only yaml and json files are supported to save reduced values.'
        assert (output_reduce_file_path is not None and custom_reduce_save_callback is None) or \
               (output_reduce_file_path is None and custom_reduce_save_callback is not None) or \
               (output_reduce_file_path is None and custom_reduce_save_callback is None)
        'If the custom reduce callback has been specified the output_reduce_file_path should be None'
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count() - 1
        self.reader = reader
        self.saver = saver
        self.processing_chunk_size = processing_chunk_size
        self.tasks = []
        self.inference_tasks = []
        self.output_files_counter = 0
        self.output_rows_last_file = 0
        self.output_reduce_file_path = output_reduce_file_path
        self.output_reduce_file_format = output_reduce_file_format
        self.custom_reduce_save_callback = custom_reduce_save_callback
        self.verbose = verbose
        self._proxy_object = None

        if verbose and logger is None:
            self.logger = self._create_and_configure_logger()
        elif verbose and logger is not None:
            self.logger = logger

    def process_task(self, task: AbstractProcessingTask, inference=False) -> DataPipeline:
        """
        Adds a new processing task
        """
        assert task.type() == TaskType.PROCESSOR
        self.tasks.append(task)
        if inference:
            self.inference_tasks.append(task)
        return self

    def filter_task(self, task: AbstractFilterTask, inference=False) -> DataPipeline:
        """
        Adds a new filtering task
        """
        assert task.type() == TaskType.FILTER
        self.tasks.append(task)
        if inference:
            self.inference_tasks.append(task)
        return self

    def reduce_task(self, task: AbstractFilterTask) -> DataPipeline:
        """
        Adds a new reducing task
        """
        assert task.type() == TaskType.REDUCER
        self.tasks.append(task)
        return self

    def balance_task(self, task: AbstractBalancingTask) -> DataPipeline:
        """
        Adds a new balancing task
        """
        assert task.type() == TaskType.BALANCING
        self.tasks.append(task)
        return self

    def flatten(self) -> DataPipeline:
        """
        Adds a new flattened task
        """
        if len(self.tasks) > 0 and all([t.type() != TaskType.PROCESSOR for t in self.tasks]):
            raise Exception(self._FLATTEN_ORDER_EXCEPTION)
        self.flattened_steps.append(len(self.tasks))
        return self

    def proxy_object(self, proxy_object):
        self._proxy_object = proxy_object
        return self

    def _get_reduce_tasks(self):
        return [task for task in self.tasks if task.type() == TaskType.REDUCER]

    def _create_and_configure_logger(self):
        logger = logging.getLogger('datashift')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
        return logger

    def _calculate_subcategories_probabilities(self, data_list, task: AbstractBalancingTask) -> dict:
        cat_and_subcat_dict = self._retrieve_and_validate_distribution_categories(data_list, task)
        for key in cat_and_subcat_dict.keys():
            cat_and_subcat_dict[key] = {k: v for k, v in
                                        sorted(cat_and_subcat_dict[key].items(), key=lambda item: item[0])}
        adjusted_cat_and_subcat_dict = self._adjust_number_of_samples_per_subcategory(cat_and_subcat_dict,
                                                                                      task.max_proportion_difference_characteristic,
                                                                                      task.characteristic_distribution)
        adjusted_cat_and_subcat_dict = self._adjust_number_of_samples_per_category(adjusted_cat_and_subcat_dict,
                                                                                   task.max_proportion_difference_category)
        probabilities = {}
        for cat_name, subcategories in cat_and_subcat_dict.items():
            if cat_name not in probabilities:
                probabilities[cat_name] = {}
            for subcat_name, n_samples in subcategories.items():
                probabilities[cat_name][subcat_name] = adjusted_cat_and_subcat_dict[cat_name][subcat_name] / \
                                                       cat_and_subcat_dict[cat_name][subcat_name]
        return probabilities

    def _adjust_number_of_samples_per_category(self, cat_and_subcat_dict, max_proportion_difference_category) -> dict:
        min_cat = min(
            [sum(subcat.values()) for subcat in cat_and_subcat_dict.values()]) * max_proportion_difference_category
        for cat, subcat in cat_and_subcat_dict.items():
            to_remove = sum(subcat.values()) - min_cat

            # make sure that _remove_from_values() uses sorted subcategories to preserve order in each chunk
            subcat_sorted = {k: v for k, v in sorted(subcat.items(), key=lambda item: item[1], reverse=True)}

            new_values = self._remove_from_values(list(subcat_sorted.values()), to_remove)
            for k, v in zip(subcat_sorted.keys(), new_values):
                cat_and_subcat_dict[cat][k] = v
        return cat_and_subcat_dict

    def _remove_from_values(self, values, n_to_remove) -> list:
        # for categories, where total numbers of samples < (min_cat * max_proportion_difference_category)
        # n_to_remove is negative, thus don't do any adjustments
        if n_to_remove < 1:
            return values
        else:
            # else remove percent of samples proportionally for every subcategory
            percent = n_to_remove / sum(values)
            new_values = []
            for value in values:
                new_values.append((1 - percent) * value)
            return new_values

    def _adjust_number_of_samples_per_subcategory(self, cat_and_subcat_dict,
                                                  max_proportion_difference_subcategory=None,
                                                  characteristic_distribution=None) -> dict:
        adjusted_cat_and_subcat_dict = {}
        for cat, subcat in cat_and_subcat_dict.items():
            adjusted_cat_and_subcat_dict[cat] = {}
            if characteristic_distribution is None:
                min_subcat = min(subcat.values()) * max_proportion_difference_subcategory
                for subcat_name in subcat.keys():
                    if subcat[subcat_name] > min_subcat:
                        adjusted_cat_and_subcat_dict[cat][subcat_name] = min_subcat
                    else:
                        adjusted_cat_and_subcat_dict[cat][subcat_name] = subcat[subcat_name]
            else:
                to_be_characteristic = {k: v * sum(subcat.values()) for k, v in characteristic_distribution.items()}
                correction = min([subcat[k] / to_be_characteristic[k] for k in subcat.keys()])
                for subcat_name in subcat.keys():
                    adjusted_cat_and_subcat_dict[cat][subcat_name] = int(to_be_characteristic[subcat_name] * correction)
        return adjusted_cat_and_subcat_dict

    def _setup_tasks(self):
        for task in self.tasks:
            task.setup()

    def _excepthook(self, exctype, value, traceback):
        traceback.print_exc()
        self.logger.error("Type: {}".format(exctype))
        self.logger.error("Value: {}".format(value))
        self.logger.error("Traceback: {}".format(traceback))
        for p in multiprocessing.active_children():
            p.terminate()

    def _execute_process(self, input_data):
        try:
            return self._execute_pipeline(input_data)
        except Exception as e:
            self.logger.error("Catching exception...")
            self.logger.error(e)
            traceback.print_exc()
            raise Exception(e.message)

    def _execute_pipeline(self, input_data) -> list:
        start_time = time.time()
        local_reductions_file_mapping = {}
        data_bucket, tmp_dir, proxy_object = input_data
        data_bucket.setup()
        self._setup_tasks()
        for task in self.tasks:
            task.assign_proxy_object(proxy_object)
        self._print_logs('Starting processing of {}'.format(data_bucket))
        data_list = data_bucket.next_data_chunk()
        while data_list is not None and len(data_list) > 0:
            local_reductions = {}
            assert len(self.tasks) > 0
            for t_iter, task in enumerate(self.tasks):
                if task.type() == TaskType.REDUCER and len(data_list) > 0:
                    self._validate_and_reduce_locally(task, data_list, local_reductions)
                else:
                    if task.type() == TaskType.BALANCING:
                        balancing_probabilities = self._calculate_subcategories_probabilities(data_list, task)
                    elements = []
                    for data in self.gen_chunks(data_list,
                                                task.get_chunk_size()) if task.get_chunk_size() > 1 else data_list:
                        if task.type() == TaskType.PROCESSOR:
                            elements.append(task.process(data))
                        elif task.type() == TaskType.FILTER and task.filter(data):
                            elements.append(data)
                        elif task.type() == TaskType.BALANCING:
                            selected_categories = self._calculate_if_given_sample_should_be_selected(data, task,
                                                                                                     balancing_probabilities)
                            if selected_categories:
                                task.mark_sample_as_selected(data, selected_categories)
                                elements.append(data)

                    if t_iter + 1 in self.flattened_steps:
                        data_list = [item for sublist in elements for item in sublist]
                    else:
                        data_list = elements
            if len(data_list) > 0 and self.saver is not None:
                self.saver.save(data_list)
            if len(local_reductions) > 0:
                for reduced_value_name, value in local_reductions.items():
                    fp = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
                    pickle.dump(value, fp, protocol=pickle.HIGHEST_PROTOCOL)

                    # local reduction tmp file was overwritten with multiple chunks in single execution
                    # solution: return metadata tmp files as list, not single path
                    if reduced_value_name in local_reductions_file_mapping:
                        local_reductions_file_mapping[reduced_value_name].append(fp.name)
                    else:
                        local_reductions_file_mapping[reduced_value_name] = [fp.name]
                    fp.close()
            data_list = data_bucket.next_data_chunk()
        data_bucket.teardown()
        self._teardown_tasks()
        self._print_logs("Process {} - Finished data bucket in {}s".format(os.getpid(), time.time() - start_time))
        return local_reductions_file_mapping

    def _teardown_tasks(self):
        for task in self.tasks:
            task.teardown()

    def _calculate_if_given_sample_should_be_selected(self, sample, task, balancing_probabilities) -> list:
        selected_categories = []
        for category in task.determine_categories(sample):
            if np.random.uniform() <= balancing_probabilities[category][
                task.determine_characteristic(sample)]:
                selected_categories.append(category)
        return selected_categories

    def _calculate_min_occurences(self, dist_categories, max_proportion_difference) -> float:
        return min(dist_categories.values()) * max_proportion_difference

    def _retrieve_and_validate_distribution_categories(self, data_list, task: AbstractBalancingTask) -> dict:
        cat_and_subcat = {}
        for data in data_list:
            distribution_categories = task.determine_categories(data)
            if type(distribution_categories) != list or not distribution_categories or any(
                    [type(c) != str for c in distribution_categories]):
                raise TypeError(
                    'The balancing task should return distribution categories as a list of str, not {}.'.format(
                        type(distribution_categories)))
            distribution_subcategory = task.determine_characteristic(data)
            if type(distribution_subcategory) != str:
                raise TypeError('The balancing task should return distribution subcategory as str, not {}'.format(
                    type(distribution_subcategory)))
            for category in distribution_categories:
                if category not in cat_and_subcat:
                    cat_and_subcat[category] = {distribution_subcategory: 1}
                elif distribution_subcategory not in cat_and_subcat[category]:
                    cat_and_subcat[category][distribution_subcategory] = 1
                else:
                    cat_and_subcat[category][distribution_subcategory] += 1
        return cat_and_subcat

    def _validate_and_reduce_locally(self, task, data_list, local_reductions) -> dict:
        reduced_locally_value = task.reduce_locally(data_list)
        if type(reduced_locally_value) not in (float, int, str, dict):
            raise TypeError("Final value of the reduced chunk must be dict, float, int or str, not {}.".format(
                type(reduced_locally_value)))
        local_reductions[task.reduced_value_name] = reduced_locally_value

    def _chunk_by_n_parts(self, data_list, num) -> list:
        avg = len(data_list) / float(num)
        out = []
        last = 0.0
        while last < len(data_list):
            out.append(data_list[int(last):int(last + avg)])
            last += avg
        return out

    def gen_chunks(self, reader, chunksize) -> Iterator[list]:
        """
        Chunk generator. Take a CSV `reader` and yield
        `chunksize` sized slices.
        """
        chunk = []
        for i, line in enumerate(reader):
            if (i % chunksize == 0 and i > 0):
                yield chunk
                del chunk[:]
            chunk.append(line)
        yield chunk

    def _reduce_globally(self, reduction_task) -> tuple:
        task, local_reductions_file_names = reduction_task

        def next_local_reduction_gen():
            for file_name in local_reductions_file_names:
                with open(file_name, 'rb') as fp:
                    yield pickle.load(fp)

        result = task.reduce_globally(next_local_reduction_gen)
        if type(result) not in (float, int, str, dict):
            raise TypeError("Reduced values can be only dict, str, float or int, not {}.".format(type(result)))
        return task.reduced_value_name, result

    def _save_reduced(self, global_reductions) -> None:
        dict_to_save = {}
        for k, v in global_reductions:
            dict_to_save[k] = v
        if self.custom_reduce_save_callback is None:
            Path(self.output_reduce_file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_reduce_file_path, 'w') as fp:
                if self.output_reduce_file_format == 'yaml':
                    yaml.dump(dict_to_save, fp)
                elif self.output_reduce_file_format == 'json':
                    json.dump(dict_to_save, fp)
                else:
                    raise Exception("File extension {} not supported".format(self.output_reduce_file_format))
        else:
            self.custom_reduce_save_callback(dict_to_save)

    def _execute(self, tmp_dir):
        sys.excepthook = self._excepthook
        self._print_logs('Dataset shifting has started - {} workers.'.format(self.num_workers))
        if len(self._get_reduce_tasks()) == 0 and (
                self.output_reduce_file_path is not None or self.custom_reduce_save_callback is not None):
            raise AssertionError(
                "You have defined a file name or callback for reduce output but there is no task to reduce.")
        if len(self._get_reduce_tasks()) > 0 and self.output_reduce_file_path is None and self.custom_reduce_save_callback is None:
            raise AssertionError(
                "You have defined {} tasks to reduce but a file name and callback for reduce output is still not defined.".format(
                    len(self._get_reduce_tasks())))
        if self.saver is None and len(self._get_reduce_tasks()) == 0:
            raise AssertionError(
                "Please add a saver to save data or/and reduction tasks and reduction output file to reduce data.")
        pool = Pool(self.num_workers)
        self._print_logs(
            'Data scanning and generation of data chunks for multi-threaded execution. This may take a while...')
        data_buckets = self.reader.build_data_buckets(pool, self.processing_chunk_size)
        self._print_logs('Created {} dedicated data buckets for multi-threaded execution.'.format(len(data_buckets)))
        self._print_logs('Processing has started...')
        try:
            local_reductions_file_mappings = pool.map(self._execute_process,
                                                      [(db, tmp_dir, self._proxy_object) for db in data_buckets])
        except Exception as e:
            pool.terminate()
            self.logger.error('Ann error during processing occured: {}'.format(str(e)))
            raise e
        if self.saver is not None:
            self._print_logs('Processing completed.')
        if len(self._get_reduce_tasks()) > 0:
            self._print_logs('Metadata generation has started...')
            local_reductions_file_mappings = [lr for lr in local_reductions_file_mappings if
                                              lr is not None and len(lr) > 0]
            # changed list comprehension tu use path lists instead of single string path
            global_reduction_tasks = [
                (task, [k for lr in local_reductions_file_mappings for k in lr[task.reduced_value_name]]) for
                task in self._get_reduce_tasks()]
            global_reductions = pool.map(self._reduce_globally, global_reduction_tasks)
            self._save_reduced(global_reductions)
            self._print_logs(
                'Metadata generation completed. Statistics saved to {}.'.format(self.output_reduce_file_path))
        self._print_logs('Data from SUCCESSFULLY shifted!')
        pool.close()

    def shift(self) -> None:
        """
        Performs processing of the dataset according to the created pipeline.
        """
        tmp_dir = tempfile.TemporaryDirectory()
        try:
            self._execute(tmp_dir.name)
        finally:
            tmp_dir.cleanup()
            if self.saver:
                self.saver.clean_savings_statuses()

    def inference(self, data):
        for inference_task in self.inference_tasks:
            if inference_task.type() == TaskType.PROCESSOR:
                data = inference_task.process(data)
            elif inference_task.type() == TaskType.FILTER and not inference_task.filter(data):
                return None
            else:
                raise Exception('Unsupported task for inference {}.'.format(inference_task.type()))
        return data

    def _print_logs(self, message):
        if self.verbose:
            self.logger.info(message)
