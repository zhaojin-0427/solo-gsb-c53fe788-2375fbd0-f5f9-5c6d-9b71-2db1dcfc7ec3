"""pytest 配置：为测试进程隔离数据库并把仓库根目录加入 sys.path。"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="uncert-pytest-")
os.environ.setdefault("UNCERTAINTY_DB_PATH", os.path.join(_TMP, "test.db"))
