import unittest
from unittest.mock import Mock

from stock_papi.repositories.gcs import get_allowed_object


class GcsRepositoryTests(unittest.TestCase):
    def test_reader_allows_only_declared_product_prefixes(self):
        allowed = (
            "quant/v1/",
            "reports/v1/",
            "reports/v2/",
            "dashboard/v1/",
            "predictions/v1/",
            "previews/",
        )
        for prefix in allowed:
            with self.subTest(prefix=prefix):
                response = Mock(
                    status_code=200,
                    headers={},
                    iter_content=Mock(return_value=[b"ok"]),
                )
                get = Mock(return_value=response)
                self.assertEqual(
                    get_allowed_object(
                        f"{prefix}object",
                        2,
                        prefix,
                        bucket="safe-bucket",
                        enabled=True,
                        token_provider=lambda: "token",
                        http_get=get,
                    ),
                    b"ok",
                )
                response.close.assert_called_once()

    def test_reader_requires_allowlisted_prefix_and_enforces_stream_limit(self):
        response = Mock(status_code=200, headers={}, iter_content=Mock(return_value=[b"ab", b"cd"]))
        get = Mock(return_value=response)
        kwargs = {
            "bucket": "safe-bucket",
            "enabled": True,
            "token_provider": lambda: "token",
            "http_get": get,
        }

        self.assertEqual(
            get_allowed_object("quant/v1/object", 4, "quant/v1/", **kwargs),
            b"abcd",
        )
        self.assertIsNone(
            get_allowed_object("reports/v1/object", 4, "quant/v1/", **kwargs)
        )
        self.assertIsNone(
            get_allowed_object("secret/object", 4, "secret/", **kwargs)
        )
        self.assertIsNone(
            get_allowed_object("quant/v1/object", 4, [], **kwargs)
        )
        self.assertIsNone(
            get_allowed_object("quant/v1/object", 3, "quant/v1/", **kwargs)
        )
        self.assertEqual(response.close.call_count, 2)
        self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
