# Semantic UV Organizer — 日本語実装指示書

版: 1.2 / 作成日: 2026-11-10 / 状態: 開発・受入の実施指示

規範文書: [設計仕様書](DESIGN_SPEC.md)。本書は実装成果物ではなく、実装者が満たすべき作業と検証を定義する。

## 1. 実装者への指示

機械学習を用いないBlender Extensionを実装する。既定の初期対象はBlender 4.5 LTS。PythonとBlender標準APIを優先し、API差異は互換層に集約する。Blender 5.xや追加OSへの対応は実機検証後に宣言する。

R-001〜R-008、確定8条件R-101〜R-108、部分再展開R-201〜R-206を必須とする。作業優先順位は元データの保護、正しい展開、シーム品質、意味的配置、充填効率。部分再展開とUDIMを完成版から省略しない。

既存UVの上書き、オブジェクト結合、無断共有Mesh独立化、対称UVスタック、無断UDIM化、未選択面の修正、ハード制約違反結果のApplyは禁止する。マテリアル統合やベイクを追加しない。

## 2. モジュール構成

```text
semantic_uv/
  __init__.py
  blender_manifest.toml
  core/{models,settings,pipeline,errors}.py
  blender/{mesh_adapter,uv_adapter,compatibility,context}.py
  analysis/{topology,curvature,materials,symmetry,shape_features}.py
  seams/{scoring,constraints,segmentation,optimization}.py
  unwrap/{backend,distortion}.py
  partial/{selection,boundaries,constraints,pipeline}.py
  layout/{similarity,grouping,orientation,texel_density,domain,obstacles,packing,optimization}.py
  validation/{uv_validity,overlaps,invariants,quality}.py
  persistence/{snapshot,transaction,restore,serialization}.py
  ui/{operators,panels,properties,overlays}.py
  tests/
    unit/
    blender_integration/
    fault_injection/
    fixtures/
    benchmarks/
```

純粋な解析・配置関数はBlenderをimportせず入力配列から結果を返す。Blenderデータへの読み書きはアダプターとトランザクションへ限定する。UIオペレーターに最適化ロジックを書かない。

## 3. 設定と関数契約

最低限以下を型付きモデルとして定義する。

| 設定 | 値・初期値 |
|---|---|
| scope | ALL_FACES / SELECTED_FACES |
| uv_mode | GENERATE / IMPROVE |
| boundary_mode | KEEP_BOUNDARY（既定）/ RELOCATE_SELECTED |
| seam_mode | PROTECT / PREFER / IGNORE |
| tile_mode | SINGLE（既定）/ UDIM |
| allowed_tiles | SINGLEでは[1001]、UDIMでは明示集合 |
| resolution / gap_px | 初期2048 / 8、タイルは同一解像度 |
| density_mode | 全体: 共通倍率を領域に適合、部分再配置: 指定値または固定領域から推定 |
| preset | Balanced / Texture Painting / Compact Packing |
| search | 固定候補数、反復数、シード、安全用時間・メモリ上限 |
| shared_mesh_policy | 既定STOP、明示許可時のみMAKE_SINGLE_USER |

APIの推奨契約は以下。名称変更時も責務と不変条件は維持する。

```python
capture_input(context, settings) -> InputSnapshot
validate_input(snapshot, settings) -> ValidationReport
build_scope(snapshot, settings) -> PartialScope | FullScope
build_domain(snapshot, scope, settings) -> LayoutDomain
compute_plan(snapshot, scope, domain, settings) -> ResultPlan
validate_plan(snapshot, plan, settings) -> ValidationReport
commit_plan(snapshot, plan) -> TransactionRecord
rollback(record) -> RestoreReport
restore(record, current_state) -> RestoreReport
```

ResultPlanはBlender参照を含まない。全UVコーナーの候補と書込許可集合、入力指紋、品質、候補シーム、警告、探索履歴を持つ。エラーは型付きコードと対象IDを返す。例外を握りつぶして成功扱いしない。

## 4. 開発フェーズと完了ゲート

| フェーズ | 実装内容 | 完了条件 |
|---|---|---|
| P0 | Extension骨格、ID、スナップショット、共有Mesh検出、永続バックアップ、一括確定／復元 | 計算で入力不変。各書込点の異常から全対象復元。保存再読込後Restore |
| P1 | 接続、ワールド面積、曲率、マテリアル、非多様体／退化検出 | 基本形状と非一様スケールで特徴量を検証 |
| P2 | シーム候補、保護／優先／無視、REQUIRE/FORBID、Chart分割 | ハード制約衝突で停止。候補生成に再現性 |
| P3 | 新規生成、既存改善、標準展開バックエンド、歪み／重複評価 | 全体の有効な初期UV。元UVを保持 |
| P4 | 部分範囲、境界、pin、KEEP_BOUNDARY | T-201〜T-205、T-212〜T-214の該当項目合格 |
| P5 | 類似形状、対称性、グループ化、方向、密度 | 独立配置とワールド面積比を検証 |
| P6 | 共通パッキング、UDIM、固定障害物、RELOCATE_SELECTED | 複数対象の空き領域配置。T-206〜T-211、T-215〜T-216合格 |
| P7 | UI、プレビュー、比較、取消、設定とエラー表示 | 全ワークフローをUIから実行し、取消後入力不変 |
| P8 | 統合、異常注入、性能、配布 | 全受入合格、閾値確定、既知制限・互換性を記録 |

各フェーズで実装内容、実行環境、試験結果、既知制限、次フェーズへの影響を報告する。P4/P6を後日の任意機能にしない。

## 5. Blender境界の実装ルール

1. Edit Modeでは編集データを整合した方法で読み、モード・アクティブObject・Object／面選択・UV選択を保存する。処理後に戻す。
2. BMeshのインデックス更新後にスナップショットを作り、処理中のトポロジー変更を禁止する。UVはloop／面コーナー単位で扱う。
3. `bpy.ops`は作業コピーだけに作用させ、アクティブObject、モード、選択、UVレイヤー、pin、コンテキストを明示する。実行可否と返値を検証する。
4. 作業コピーでのUVマージや選択連動が元データに影響しないことを試験する。処理後は作成した一時資源だけを解放する。
5. BlenderデータAPIはメインスレッドで操作する。長処理は段階化して取消を受け付ける。バックグラウンド計算へBlenderオブジェクトを渡さない。
6. 未適用変換を変更せず、三角形の頂点をワールド変換して面積を求める。内部三角形化で元の面構成を変更しない。
7. UV Operatorのpinや部分選択の挙動を、対象バージョンのテストで確認する。挙動が不十分なら制約付き解法を補うか、説明付き失敗とする。検証を省略して対応済みとしない。

## 6. 部分再展開アルゴリズム

### 6.1 スコープ構築

対象Objectごとに、起動時の面選択からSとUを作る。隠れ面はU。UVコーナーの集合を分離し、S/U隣接エッジとメッシュ外周から境界を抽出する。UVの同値判定は既存UVの実際の連続性で行い、同じ頂点だからという理由でコーナーを溶接しない。

KEEP_BOUNDARYはS側境界コーナーと内部pinを固定する。Uはすべて固定。RELOCATE_SELECTEDはS側境界の位置固定のみを解除するが、U、明示pin、FORBID等は解除しない。境界分離がFORBIDと衝突すればCONSTRAINT_CONFLICT。

### 6.2 処理の擬似コード

```text
snapshot = capture_input()
validate_input_or_stop(snapshot)
scope = build_selected_face_scope(snapshot)
if scope.selected_face_count == 0: return NO_OP
candidate = exact_copy_of_source_uv(snapshot)
constraints = build_boundary_and_pin_constraints(scope)
domain = build_common_tile_domain_and_fixed_obstacles(snapshot, scope)

for region in stable_order(scope.selected_components):
    charts = generate_or_improve_selected_charts(region, constraints)
    uv = unwrap_on_working_copy(charts, constraints)
    if KEEP_BOUNDARY:
        verify_fixed_boundary(uv)
    else:
        match_density_without_moving_hard_pins(uv)
        place_into_free_domain(uv, domain)
        reserve_placed_geometry(domain, uv)
    candidate.write_only_selected_corners(uv)

verify_unselected_corners_exact(snapshot, candidate)
verify_all_objects_and_hard_constraints(candidate)
preview(candidate)
on_explicit_apply:
    revalidate_input_fingerprint()
    atomic_commit_all_objects_to_new_uv_layers(candidate)
```

局所探索も書込許可集合の外へ出てはならない。解けない境界を削除したり、近接面まで自動選択したりしない。固定座標はソルバーが許容誤差内で返しても最後に元の数値を代入し直し、その最終値で重複・歪みを再検証する。

### 6.3 障害物とパッキング

共通セット全対象のUをUV三角形として収集し、AABB／空間インデックスで候補を絞る。Sの旧位置だけを障害物から外す。S/Uの共有頂点で障害物が削除されないことを確認する。

RELOCATE_SELECTEDは有限の回転候補と安定した島順で空き領域を探索する。初期矩形配置の失敗後、輪郭に基づく限定探索を行う。各確定候補は実三角形／輪郭で衝突と余白を検査する。既存Uは移動、回転、拡縮しない。新たな島も配置後に障害物へ加える。密度を勝手に落として適合させない。

KEEP_BOUNDARYで連続しているS/Uの共通エッジは正規の接触として許容する。これを正面積重複や島間余白不足と誤認しない。非隣接面の貫通・重複は検出する。

## 7. 密度、品質、決定性の実装

`A_world`はワールド三角形面積の和、`A_uv`はUV三角形面積の和。倍率は`sqrt(k_target/k_current)`。固定U由来の代表値は、有効な固定アイランドの`A_uv`合計／`A_world`合計。Uの密度偏差も表示し、Sだけの調整で全体統一したと誤表示しない。

初期の数値設定案を以下とし、P1の数値試験で確定する。設定値と単位はテストレポートに記録する。

| 項目 | 初期案／判定 |
|---|---|
| 未選択／元UV不変 | 座標配列を要素単位で完全一致。許容差で変更を隠さない |
| 固定境界・pin | 最終値は元値へ復帰し完全一致 |
| 幾何比較epsilon | UV長さ1e-7を起点にスケール適応。UDIMはタイルローカル座標で計算 |
| 重複面積epsilon | タイルローカル面積1e-12を数値誤差起点とし、面の大きさに応じて検証 |
| 密度偏差 | 制約のない全体配置／再配置で島ごと目標±5%を初期受入値 |
| 島間距離 | 8px/2048px。数値誤差を除いて不足を許容しない |
| 決定性 | 同一Blenderビルド・OS・設定・シード・候補予算で同一出力配列 |

epsilonを実重複や極小の不正面を無視する抜け道にしない。退化はスケール相対判定し、有限性を最初に検査する。KEEP_BOUNDARYの密度偏差はハード不合格ではなく制約付き警告とし、例外の対象と理由を明示する。

三角形対応から角度歪みと面積歪みを計算し、品質スコアを正規化する。正常な既存UVより総合コストが悪い改善候補は棄却する。固定候補順、IDでのタイブレーク、固定シードを用い、ハッシュ集合の列挙順に依存しない。時間上限で中断した場合は使用済み候補数と最後の有効解を記録する。

## 8. トランザクションと復元

P0で次の順序を完成させ、後続処理から直接元メッシュを書き換えない。

1. 入力・コンテキストを取得し、全対象の事前検証を行う。
2. 全変更対象のバックアップを取得し、復元可能性を検証する。
3. 作業コピーで計算し、全体検証済みResultPlanをプレビューする。
4. Apply時に全入力指紋、編集可能性、出力レイヤー容量、共有参照を再検証する。
5. 永続TransactionRecordを準備し、作成したレイヤー・データ・参照変更を順に記録する。
6. 全対象へ結果を反映して事後不変検証を行い、成功したときだけCOMMITTEDにする。
7. 任意の例外・取消では逆順に戻し、全対象を再検査する。ロールバック中はユーザー取消で中断しない。

Restoreは元の全UV・属性・レイヤー設定・シーム・参照を復元し、今回作成した結果と独立化データのみを安全に除去する。共有参照の再接続前に後続変更と新規ユーザーを確認する。全対象の競合確認を復元開始前に済ませ、競合時は元状態を壊さず停止する。復元データはバージョン付き形式で.blend保存に含め、再読込後の復元を試験する。

Undoの有無とは独立に試験する。強制終了中の完全原子性は保証せず、保存済みバックアップから回収可能な範囲を利用説明に記載する。

## 9. 受入テスト

### 9.1 共通手順とフィクスチャ

Cube、UV Sphere、Cylinder、Torus、Suzanne、複数マテリアルCube、対称機械部品、独立成分、多Object、非多様体、退化面、高ポリゴンを用意する。部分用に、連続グリッドの中央面群、外周を含む面群、穴のある選択、複数非連結選択、pin付き島、既存スタック、空きタイルを持つモデルを追加する。

各試験で入力全UV、シーム、形状、マテリアル、変換、モード、選択の基準を保存する。成功結果と失敗後状態を配列比較し、オブジェクト横断で検証する。見た目のスクリーンショットだけで合否を決めない。

### 9.2 確定8条件に対応する既存テスト

| ID | 入力／操作 | 合格条件 |
|---|---|---|
| T-101 | 3個以上の独立Objectを全体展開 | 結合せず共通UVに配置、Object間重複ゼロ |
| T-102 | 既存UV付きモデルでGENERATE | 新規UVに有効結果、全既存UVは完全一致 |
| T-103 | IMPROVE | 元UVを初期解に使用し、新規レイヤーに出力。正常基準から総合品質悪化なし |
| T-104 | 同形状、非一様／未適用スケール違い | ワールド面積に比例し密度偏差±5%以内 |
| T-105 | PROTECTと手動REQUIRE/FORBID | 保護制約違反ゼロ。元シーム不変 |
| T-106 | PREFERで改善に切断変更が必要な例 | 変更コストを考慮し必要箇所だけ変更、理由を報告 |
| T-107 | IGNORE、異なる既存ソフトシーム | 同じハード制約・幾何なら既存ヒントに依存せず候補生成 |
| T-108 | 対称部品 | 独立配置、意図的重複ゼロ、対応方向を比較表示 |
| T-109 | SINGLE | 全島が1001内、端余白を満たす |
| T-110 | UDIM指定1001/1002/1011 | 許可タイルだけを使用、境界横断なし、番号対応と共通衝突検査合格 |
| T-111 | 2個目のObjectの確定時に例外注入 | 1個目を含む全対象を復元、残存出力なし |
| T-112 | 同一環境・固定探索予算で再実行 | 出力UV配列・タイル・グループが完全一致 |

### 9.3 部分再展開の必須テスト

| ID | 入力／操作 | 合格条件 |
|---|---|---|
| T-201 | 連続グリッドの中央面だけ選択 | Sだけ再展開し、Uの全コーナー座標・pin・UV選択属性が完全一致 |
| T-202 | KEEP_BOUNDARY、S/U連続境界 | S側とU側の境界座標が完全一致で不変。内部に有効結果 |
| T-203 | 既存UVシームで分かれた境界、共有3D頂点 | コーナーを混同せず、両側の元境界座標を保持 |
| T-204 | 選択内部pin、外周、穴、非連結面群 | 各境界とpinを保持。選択範囲の自動拡張なし |
| T-205 | 位置固定では解けない入力 | 境界解除や再配置へ自動切替せず失敗、全入力不変 |
| T-206 | RELOCATE_SELECTED、Uの間に十分な空き | S側だけ境界分離し、空きへ移動。U不変、余白・密度達成 |
| T-207 | Aの選択島とBの固定UVが衝突する初期案 | Bも障害物として避け、全Object間重複ゼロ |
| T-208 | A/B双方に可動選択島 | 先行配置を障害物登録し、可動島同士も重ならない |
| T-209 | 許可タイルに空き不足 | NO_FEASIBLE_LAYOUT、縮小・U移動・無断UDIM化なし、全対象ロールバック |
| T-210 | UDIM部分配置、1002のみ空き | 許可時のみ1002へ配置、固定UVのタイル・座標不変 |
| T-211 | 再配置境界にFORBID、または位置pinで移動不能 | 制約を外さず、説明付き失敗。実シーム不変 |
| T-212 | KEEP_BOUNDARYで固定面との正規の接触 | 正面積重複と誤検出せず、連続境界に余白を強要しない |
| T-213 | UVなし／選択ゼロ／一部Objectのみ選択ゼロ | UVなしは停止、全体選択ゼロはno-op。選択ゼロObjectのUVは障害物かつ書込なし |
| T-214 | 固定Uに重複・NaN・退化、許可外タイル | Apply不可、対象面と理由を示しUを修復しない |
| T-215 | Sの旧領域とUが一部重なる障害物構成 | Sの旧位置解放でUの障害物を削除しない |
| T-216 | 部分計算中取消、部分Apply途中失敗、部分結果Restore | 全対象で元UV・属性・コンテキストへ復帰。結果レイヤーも適切に除去 |

### 9.4 安全性、品質、配布テスト

| ID | 入力／操作 | 合格条件 |
|---|---|---|
| T-301 | 共有Meshの対象／非対象インスタンス | 既定停止。明示独立化時のみ独立出力、非対象不変、Restoreで参照復帰 |
| T-302 | Preview後にトポロジー・UV・選択・変換を変更 | STALE_INPUTでApply拒否、古い面IDへ書込なし |
| T-303 | Apply後に結果を手動変更してRestore | RESTORE_CONFLICT、手動編集を上書きしない |
| T-304 | Apply、.blend保存、閉じる、再読込、Restore | 永続バックアップから全対象を正しく復元 |
| T-305 | 全書込段階に例外、取消、容量不足を注入 | 復元、失敗状態表示、一時資源清掃。復元失敗の模擬では残存状態を正直に報告 |
| T-306 | 非多様体・退化・ゼロスケール・読取専用 | 問題箇所を報告し安全停止。無断修復・部分確定なし |
| T-307 | オフライン、新規インストール、登録解除 | ネットワーク／ML依存なし。対象OSとBlenderの実測結果を記録 |
| T-308 | 改善／新規／部分の品質・性能比較 | 定義済み閾値内、歪みと可読性指標を併記し、未測定値を保証しない |

## 10. ベンチマークとリリース判定

1万、10万、50万三角形、Object数1／10／100を基本規模とし、代表モデルに全体・部分、単一・UDIMを組み合わせる。CPU、RAM、OS、Blenderビルド、入力ハッシュ、設定、候補予算、処理段階別時間、最大メモリ、取消応答時間を記録する。

Smart UV Projectは同じ入力・タイル・余白・解像度条件でベースライン比較に用いる。すべてのモデルで優越する必要はない。人間による部品対応の把握、描画・修正作業の評価も併記し、代理スコアだけで可読性を断定しない。

P1で数値誤差基準、P8前に平均／最大歪み、処理時間、メモリ、取消応答、最低品質の製品閾値を測定して固定する。測定条件・基準モデル・合格値を版管理し、未確定のままT-308を合格にしない。性能上限超過は説明付き安全停止とする。

リリースにはExtension一式、利用手順、設定説明、Restore／回収手順、全受入結果、ベンチマーク、互換性一覧、既知制限を含める。安全性・部分再展開・複数Object・UDIMの必須試験に未合格が残る場合は完成版としない。無断の仕様緩和でテストを通さない。
