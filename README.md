# Semantic UV Organizer

機械学習を使わない Blender 用の自動 UV 展開・整理 Extension です。仕様は [docs/DESIGN_SPEC.md](docs/DESIGN_SPEC.md) と [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)、実装状況と試験結果は [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) にまとめています。

元の UV・シーム・マテリアルは変更せず、結果は新しい UV マップ（`SUO_Result_001` など）に書き込みます。既存のアクティブ UV／レンダー UV の設定も自動では切り替えません。

## インストール

```sh
# Extension zip を作る (Blender 4.2 以降)
blender --command extension build --source-dir semantic_uv --output-dir dist
```

Blender の「編集 > プリファレンス > エクステンション > ディスクからインストール」で `dist/semantic_uv_organizer-0.1.0.zip` を選びます。

## 使い方

3D View または UV Editor のサイドバー（N キー）に「Semantic UV」タブが追加されます。

1. 対象の Mesh Object を選択します（複数選択すると、結合せずに共通の UV 空間へ配置します）。
2. 処理範囲・生成／改善・境界モード・シームモード・密度・タイルなどを設定します。
3. **Analyze**: 入力・制約・トポロジーを検査します（データは変更しません）。
4. **Generate**: 作業コピー上で計算し、検証に合格した結果だけをプレビューします。Esc で取り消せます。UV Editor では、再計算した部分がオレンジ、固定部分がグレーで表示されます。
5. **Compare**: 入力 UV と結果の指標（歪み・密度偏差・充填率など）を並べて表示します。
6. **Apply**: 対象すべてに一括で確定します。途中で失敗した場合は、対象すべてをロールバックします。
7. **Restore**: 直近の確定を対象すべてについて取り消します。確定後に結果が手動で編集されている場合は拒否します（上書きしません）。

### 部分再展開（処理範囲: 選択面）

Edit Mode で面を選択して実行します。入力 UV が必要です。

- **境界固定**（既定）: 未選択側と選択境界のコーナーを元の座標に固定し、内部だけを再計算します。解けない場合は失敗として止まり、もう一方のモードへ自動では切り替えません。
- **選択側を分離して再配置**: 未選択側は固定したまま、選択側を UV 上で切り離し、空いている領域へ置き直します（3D メッシュは切断しません）。密度は維持し、縮小・未選択部分の移動・タイルの自動追加は行いません。

### トポロジー問題

非多様体のエッジ、法線の向きが逆の隣接面、面積0の面がある場合、既定では警告を出し、該当エッジで UV を分割して続行します（メッシュは変更しません）。設定の「トポロジー問題」を「停止」にすると、問題箇所を報告して停止します。

### 手動制約

Edit Mode でエッジを選び、パネルから `REQUIRE` / `FORBID` / `PREFER` / `AVOID` / `AUTO` を設定します。値はエッジ属性 `suo_edge_constraint` として保存されます。`REQUIRE` と `FORBID` はハード制約で、どのモードでも解除されません。

### 復元と回収

- 復元情報は `.blend` 内の Text データブロック（`.SUO_TX_<id>`、フェイクユーザー付き）に保存されます。保存して開き直した後でも Restore できます。削除するまで残ります。
- Restore が拒否された場合は、「バックアップを別コピーへ回収」で Object の複製に確定前の UV を復元できます。元の Object は変更しません。
- プロセスのクラッシュや OS 障害が起きた場合の完全な原子性は保証しません。回収できるのは、保存済みの `.blend` に含まれる復元情報の範囲です。

## 開発

```text
semantic_uv/
  core/        models, settings, pipeline (関数契約), errors
  blender/     mesh_adapter, uv_adapter, compatibility, context   ← bpy 依存
  analysis/    topology, curvature, materials, symmetry, shape_features
  seams/       scoring, constraints, segmentation, optimization
  unwrap/      backend (LSCM / Tutte / Blender標準), chart_unwrap, distortion
  partial/     selection, boundaries, constraints, pipeline
  layout/      similarity, grouping, orientation, texel_density, domain, obstacles, packing, optimization
  validation/  uv_validity, overlaps, injectivity, invariants, quality
  persistence/ snapshot, transaction, restore, serialization            ← bpy 依存
  ui/          operators, panels, properties, overlays                  ← bpy 依存
  tests/       unit, blender_integration (異常注入を含む), benchmarks, fixtures
```

`core` / `analysis` / `seams` / `unwrap` / `partial` / `layout` / `validation` は Blender を import しないため、numpy だけで単体テストを実行できます。

```sh
# 単体テスト (Blender 不要)
python -m pytest semantic_uv/tests/unit

# Blender 統合テスト・異常注入テスト (配列比較で判定)
blender --background --factory-startup --python semantic_uv/tests/blender_integration/run_tests.py -- result.json

# ベンチマーク
blender --background --factory-startup --python semantic_uv/tests/benchmarks/bench.py -- bench.json "0x1,2x1,0x10"
```

## ライセンス

GNU General Public License v3.0 or later（[LICENSE](LICENSE)）。
