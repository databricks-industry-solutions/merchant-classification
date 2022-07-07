import unittest

from utils.regex_utils import *


class UdfTest(unittest.TestCase):

    def convert(self, text):
        text = re.sub(date_pattern, ' ', text)
        text = re.sub(price_regex, '', text)
        text = re.sub('(\(+)|(\)+)', '', text)
        text = re.sub('&', ' and ', text)
        text = re.sub('[^a-zA-Z0-9]+', ' ', text)
        text = re.sub('\\s+', ' ', text)
        text = re.sub('\\s+x{2,}\\s+', ' ', text)
        return text

    def test_regex(self):
        with open('tests/export.csv', 'r') as f:
            records = f.read().split('\n')
            for record in records:
                record = record.split(',')
                original = record[0]
                expected = record[1]
                self.assertEqual(self.convert(original), expected)


if __name__ == '__main__':
    unittest.main()
