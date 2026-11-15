import unittest

from store_client_e2e import result_exit_code


class ResultExitCodeTest(unittest.TestCase):
    def test_successful_workload_passes(self) -> None:
        self.assertEqual(result_exit_code(10, 0, 10, 0, 0), 0)

    def test_put_failures_fail_even_without_mismatches(self) -> None:
        self.assertEqual(result_exit_code(0, 5, 0, 0, 0), 21)

    def test_get_failures_fail_even_without_mismatches(self) -> None:
        self.assertEqual(result_exit_code(10, 0, 9, 1, 0), 21)

    def test_mismatch_has_its_own_failure_code(self) -> None:
        self.assertEqual(result_exit_code(10, 0, 9, 0, 1), 20)

    def test_no_successful_puts_fails(self) -> None:
        self.assertEqual(result_exit_code(0, 0, 0, 0, 0), 22)

    def test_no_successful_gets_fails(self) -> None:
        self.assertEqual(result_exit_code(10, 0, 0, 0, 0), 22)


if __name__ == "__main__":
    unittest.main()
