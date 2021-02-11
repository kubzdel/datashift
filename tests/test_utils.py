import re
import unittest

import numpy as np

from datashift.dataset import Dataset
from datashift.task import AbstractFilterTask, AbstractProcessingTask, AbstractBalancingTask, AbstractReduceTask


class ExampleBalancingTask(AbstractBalancingTask):
    def determine_characteristic(self, sample) -> str:
        if 'subcat' not in sample:
            sample['subcat']='subcat1' if np.random.uniform()<=0.3 else 'subcat2'
        return sample['subcat']

    def determine_categories(self,sample):
        if 'cat' not in sample:
            sample['cat']=[]
            sample['cat'].append('cat1' if np.random.uniform() <= 0.3 else 'cat2')
            sample['cat'].append('cat3' if np.random.uniform() <= 0.1 else 'cat4')
        return sample['cat']

    def mark_sample_as_selected(self,sample,selected_distribution_categories):
        sample['selected_distribution_categories'] = ';'.join(selected_distribution_categories)


class MinTextLengthFilterStrategy(AbstractFilterTask):
    def __init__(self, column_name, min_characters):
        self.column_name = column_name
        self.min_characters = min_characters

    def filter(self, data):
        return len(data[self.column_name]) > self.min_characters


class CleanTextTask(AbstractProcessingTask):
    def process(self, data):
        text = data['full_text'].lower()
        text = re.sub('[%s]' % re.escape('0123456789"#$%&\'()*+-/:;<=>?@[\\]^_`{|}~'), "", text)
        text = re.sub('\n', " ", text)
        text = re.sub('\r', "", text)
        text = re.sub(' +', " ", text)
        text = text.strip()
        data['full_text'] = text
        return data


class SplitTextTask(AbstractProcessingTask):
    def process(self, data):
        return [data, data]

class CountCategoriesTask(AbstractReduceTask):
    def __init__(self, reduced_value_name=None):
        super().__init__(reduced_value_name)

    def reduce_locally(self, chunk_samples):
        result={}
        for sample in chunk_samples:
            for c in sample['selected_distribution_categories'].split(';'):
                if c not in result:
                    result[c]=1
                else:
                    result[c]+=1
        return result

    def reduce_globally(self, reduced_chunks):
        results={}
        for k in reduced_chunks[0].keys():
            for r in reduced_chunks:
                if k not in results:
                    results[k]=r[k]
                else:
                    results[k] += r[k]
        return results

class CountSubcategoriesPerCateogry(AbstractReduceTask):
    def __init__(self, reduced_value_name,category_name):
        super().__init__(reduced_value_name)
        self.category_name=category_name

    def reduce_locally(self, chunk_samples):
        result={}
        for sample in [s for s in chunk_samples if self.category_name in s['selected_distribution_categories']]:
            if sample['subcat'] not in result:
                 result[sample['subcat']]=1
            else:
                result[sample['subcat']]+=1
        return result

    def reduce_globally(self, reduced_chunks):
        results={}
        for k in reduced_chunks[0].keys():
            for r in reduced_chunks:
                if k not in results:
                    results[k]=r[k]
                else:
                    results[k] += r[k]
        return results


class MeanValueReduceTask(AbstractReduceTask):
    def __init__(self, reduced_value_name=None):
        super().__init__(reduced_value_name)

    def reduce_locally(self, chunk_samples):
        values=[len(sample['full_text'].split()) for sample in chunk_samples]
        return sum(values)/len(values)

    def reduce_globally(self, reduced_chunks):
        return sum(reduced_chunks)/len(reduced_chunks)


class MaxValueReduceTask(AbstractReduceTask):
    def __init__(self, reduced_value_name=None):
        super().__init__(reduced_value_name)

    def reduce(self, acc, value, n):
        if acc is None:
            return value
        else:
            return max(acc, value)


class MinValueReduceTask(AbstractReduceTask):
    def __init__(self, reduced_value_name=None):
        super().__init__(reduced_value_name)

    def reduce(self, acc, value, n):
        if acc is None:
            return {'in_example': value, 'out_example': 0}
        else:
            return {'in_example': min(acc['in_example'], value), 'out_example': 0}

