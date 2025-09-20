import unittest

from src.utils.file_io import *

class TestFileIO(unittest.TestCase):

    def test_is_float(self):
        test_elements = ['1234567', '1_2_3.4', 'NaN', '-iNf', True,
                         1, '42e3', '0E0', 'infinity', '123.4567',
                         '1e424242', '1.2345678901234567890e-05', '.42', '0', '+1e1',
                         '4-2', '', '#42', '42%', '123EE4',
                         '1e5e4', '1e5^4', '1e1.1', '(5)', 'NULL',
                         '1,2', '127.28.50.00', '+-1', 'True', 'banana']
        result = [is_float(elem) for elem in test_elements]
        expected = [True] * 15 + [False] * 15
        self.assertListEqual(result, expected,
                               msg='First 15 elements should be true, rest 15 should be false')

    def test_is_int(self):
        test_elements = ['1234567890', '+1', '-1', '000005', '42',
                         '0.0', 'hello', 'NaN', '-iNf', '']
        result = [is_int(elem) for elem in test_elements]
        expected = [True] * 5 + [False] * 5
        self.assertListEqual(result, expected,
                               msg='First 15 elements should be true, rest 15 should be false')
        
    def test_is_json(self):
        test_files = ['hello.JSON', '/hello/world/hey.json', 'hello_world.JsOn',
                      'hello.txt', 'hello.JPG', 'hello.py',
                      'hello.conf', 'hello.exe', '/',
                      '', '.json', '/.json']
        result = [is_json(file) for file in test_files]
        expected = [True] * 3 + [False] * 9
        self.assertListEqual(result, expected,
                               msg='First 3 file paths should be true, rest 9 should be false')
    
    def test_parse_value(self):
        test_values = [1, 'hello', '1', 'TruE', '0.00000', ' so false', None, 'None', '42', '42.']
        test_dest_types = [list, list, bool, bool, bool, bool, list, list, int, float]
        result = [parse_value(dest_type=dest_type, value=value) for dest_type, value in zip(test_dest_types, test_values)]
        expected = [[1], ['hello'], True, True, False, False, [None], None, 42, float('42.')]
        self.assertListEqual(result, expected,
                        msg='Parse value is behaving unexpectedly, proceed carefully')
if __name__ == '__main__':
    unittest.main()