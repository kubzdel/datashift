import re

from datashift.task import AbstractFilterTask, AbstractProcessingTask, AbstractBalancingTask, AbstractReduceTask


class ExampleBalancingTask(AbstractBalancingTask):
    CATEGORIES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def determine_characteristic(self, sample) -> str:
        sample['subcat'] = 'default'
        return sample['subcat']

    def determine_categories(self, sample):
        categories = [c for c in self.CATEGORIES if sample[c] == 1]
        return categories if categories else ['other']

    def mark_sample_as_selected(self, sample, selected_distribution_categories):
        sample['selected_distribution_categories'] = ';'.join(selected_distribution_categories)


class MinTextLengthFilterStrategy(AbstractFilterTask):
    def __init__(self, column_name, min_characters):
        self.column_name = column_name
        self.min_characters = min_characters

    def filter(self, data):
        return len(data[self.column_name]) > self.min_characters


class CleanTextTask(AbstractProcessingTask):
    def process(self, data):
        text = data['comment_text'].lower()
        text = re.sub('[%s]' % re.escape('0123456789"#$%&\'()*+-/:;<=>?@[\\]^_`{|}~'), "", text)
        text = re.sub('\n', " ", text)
        text = re.sub('\r', "", text)
        text = re.sub(' +', " ", text)
        text = text.strip()
        data['comment_text'] = text
        return data


class SplitTextTask(AbstractProcessingTask):
    def process(self, data):
        return [data, data]


class CountCategoriesTask(AbstractReduceTask):
    def __init__(self, reduced_value_name=None):
        super().__init__(reduced_value_name)

    def reduce_locally(self, chunk_samples):
        result = {}
        for sample in chunk_samples:
            for c in sample['selected_distribution_categories'].split(';'):
                if c not in result:
                    result[c] = 1
                else:
                    result[c] += 1
        return result

    def reduce_globally(self, reduced_chunks):
        results = {}
        for local_reduction in reduced_chunks:
            for k in local_reduction:
                if k not in results:
                    results[k] = local_reduction[k]
                else:
                    results[k] += local_reduction[k]
        return results


class CountSubcategoriesPerCateogry(AbstractReduceTask):
    def __init__(self, reduced_value_name, category_name):
        super().__init__(reduced_value_name)
        self.category_name = category_name

    def reduce_locally(self, chunk_samples):
        result = {}
        for sample in [s for s in chunk_samples if self.category_name in s['selected_distribution_categories']]:
            if sample['subcat'] not in result:
                result[sample['subcat']] = 1
            else:
                result[sample['subcat']] += 1
        return result

    def reduce_globally(self, reduced_chunks):
        results = {}
        for local_reduction in reduced_chunks:
            for k in local_reduction:
                if k not in results:
                    results[k] = local_reduction[k]
                else:
                    results[k] += local_reduction[k]
        return results


class MeanValueReduceTask(AbstractReduceTask):
    def __init__(self, reduced_value_name=None):
        super().__init__(reduced_value_name)

    def reduce_locally(self, chunk_samples):
        values = [len(sample['comment_text'].split()) for sample in chunk_samples]
        return sum(values) / len(values)

    def reduce_globally(self, reduced_chunks):
        return sum(reduced_chunks) / len(reduced_chunks)


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
