import unittest

from resume_held_pair import exact_job_id, normalize_fixed_nodes


class ExactHeldRecoveryTests(unittest.TestCase):
    def test_single_and_equal_range(self):
        for value in ('8', '8-8'):
            self.assertEqual(normalize_fixed_nodes(f'JobId=183781 NumNodes={value} NumCPUs=1024'),
                             'JobId=183781 NumNodes=8 NumCPUs=1024')

    def test_unequal_ranges_and_ambiguous_tokens_rejected(self):
        for value in ('8-16', '8-9', '7-8', '80', '08', '(null)'):
            with self.assertRaises(RuntimeError):
                normalize_fixed_nodes(f'NumNodes={value}')
        for text in ('', 'OtherNumNodes=8', 'NumNodes=8 NumNodes=8-8'):
            with self.assertRaises(RuntimeError):
                normalize_fixed_nodes(text)

    def test_exact_job_identity(self):
        exact_job_id('JobId=183781 JobName=t25g5sc3v2', '183781')
        for text in ('JobId=183782', 'JobId=183781 JobId=183781', 'OtherJobId=183781'):
            with self.assertRaises(RuntimeError):
                exact_job_id(text, '183781')


if __name__ == '__main__':
    unittest.main()
