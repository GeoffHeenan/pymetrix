"""Reading and writing data. ``Dataset.read`` and ``Dataset.write`` call into here.

Files (Parquet, CSV, Excel) are built in. Data services and institutional data
stores connect through reader plugins (the ``pymetrix.io`` entry point), keyed by
a URI scheme such as ``sdmx://``, so the core depends on none of them.
"""

from .parquet import read_parquet, write_parquet
from .registry import Connector, columns, connectors, read, register_reader, register_writer, write
from .selection import ParquetScan, scan
from .tabular import read_csv, read_excel, write_csv, write_excel


register_reader(".parquet", read_parquet)
register_writer(".parquet", write_parquet)
register_reader(".csv", read_csv)
register_writer(".csv", write_csv)
for _ext in (".xlsx", ".xls"):
    register_reader(_ext, read_excel)
register_writer(".xlsx", write_excel)

__all__ = ["Connector", "ParquetScan", "connectors", "read", "register_reader", "register_writer", "scan", "write"]
