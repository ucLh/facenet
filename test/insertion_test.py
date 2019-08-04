"""Test insertion in sorted list"""
import unittest
from src.test_matching import insert_element, ImageFile


class TestInsertion(unittest.TestCase):
    def setUp(self):
        self.sorted_list = [1, 2, 3, 4, 5]
        # a, b, c, d, e = ImageFile('a', 1), ImageFile('b', 2), ImageFile('c', 3), ImageFile('d', 4), ImageFile('e', 5)
        # self.img_file_list = [a, b, c, d, e]

    def test_left(self):
        self.sorted_list = insert_element(1, self.sorted_list, upper_bound=5)
        self.assertEqual([1, 1, 2, 3, 4], self.sorted_list)

    def test_left2(self):
        self.sorted_list = insert_element(1, self.sorted_list, upper_bound=6)
        self.assertEqual([1, 1, 2, 3, 4, 5], self.sorted_list)

    def test_right(self):
        self.sorted_list = insert_element(4, self.sorted_list, upper_bound=5)
        self.assertEqual([1, 2, 3, 4, 4], self.sorted_list)

    def test_right2(self):
        self.sorted_list = insert_element(5, self.sorted_list, upper_bound=6)
        self.assertEqual([1, 2, 3, 4, 5, 5], self.sorted_list)

    def test_midle(self):
        self.sorted_list = insert_element(3, self.sorted_list, upper_bound=5)
        self.assertEqual([1, 2, 3, 3, 4], self.sorted_list)

    def test_out(self):
        self.sorted_list = insert_element(6, self.sorted_list, upper_bound=5)
        self.assertEqual([1, 2, 3, 4, 5], self.sorted_list)

    def test_empty(self):
        self.sorted_list = []
        self.sorted_list = insert_element(5, self.sorted_list, upper_bound=5)
        self.assertEqual([5], self.sorted_list)

    def test_not_full(self):
        self.sorted_list = [1, 2]
        self.sorted_list = insert_element(5, self.sorted_list, upper_bound=5)
        self.assertEqual([1, 2, 5], self.sorted_list)

