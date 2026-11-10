# 異常注入テスト

書込の全段階（独立化前・レイヤー作成前・書込前・書込後・事後検証）への例外注入、MemoryError、ロールバック失敗の模擬は、Blender 実機が必要なため [../blender_integration/run_tests.py](../blender_integration/run_tests.py) の `T111_*` と `T305_*` に実装しています。注入点は `semantic_uv.persistence.transaction.FAULT_HOOK` です。
