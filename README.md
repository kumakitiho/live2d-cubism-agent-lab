# Live2D Cubism Agent Lab

Live2Dモデル制作を題材に、Codexの **Skill → Script Tool → 高レベルUIマクロ → External API Bridge** を学びながら使うためのWindows向けMVPです。`live2d-one-image-vtuber` が、仕様、素材、Cubismの3工程を構造化成果物でつなぎます。完成モデルを一発生成する仕組みではありません。

## できること

- 1枚絵からmodel scope・motion level別のLive2Dパーツ計画を生成
- source画像の観測情報とユーザー意図を分けたcharacter spec生成
- 目、口、髪、体、隠れ補完と全layer metadataを管理するcanonical asset queue
- 隠れ部分を `inferred`、要確認素材を `review_required` として追跡
- `inpaint_mask` 限定の複数生成候補、品質評価、決定的ranking、レビュー後queue更新候補
- Cubismのeye、mouth、mesh、texture評価を `asset_feedback.yaml` へ変換するfeedback loop
- queueから `asset_manifest.yaml` と `layer_map.yaml` を決定的に生成
- `asset_manifest.yaml` の構造、必須部品、実ファイル、layer map整合、Cubism handoff条件を検証
- manifestからPSDレイヤー順を作るplan-only builder stub
- `character_spec.yaml` と `layer_map.yaml` のテンプレート・軽量検証
- allowlist方式の `action_plan.yaml` 検証
- Cubismのフォーカス、PSDインポート、自動メッシュ生成、保存、Undo、スクリーンショットを名前付きマクロとして計画・実行
- Cubism External APIの登録、許可状態、ドキュメント、モデルUID、パラメータ取得・設定
- UIマクロとAPI確認を1つの実行レポートへまとめる
- PSD取り込み前後を同じAPI sessionで比較し、新規DocumentUIDと取り込み後current ModelUIDを照合する
- 実機なしでのdry-run、API payloadテスト、UI操作列テスト

任意座標クリック、頂点の自由ドラッグ、中間点の手作業配置は公開コマンドにしません。PSDインポートやボタン操作は、人間へ戻すのではなく高レベルマクロで自動化します。人間確認は見た目の良否や想定外ケースの判断に限定します。

## 1枚絵からVTuber化する流れ

```text
権利確認済みsource画像 + ユーザー希望
  ↓ character-spec-generator
character_spec.yaml
  ↓ image-to-live2d-assets
canonical queue → 自動segmentation候補 → 人間レビュー → assignment plan → queue candidate
  ↓ soft target mask抽出 → AA edge補修 / binary inpaint穴補完
generated/parts/*.png
  ↓ draw order再合成 → source差分 → quality gate → motion stress preview
失敗partだけrefinement queueへ戻す
  ↓ 生成inpainting backend（mock + optional Diffusers / FLUX Fill、redrawは未接続）
  ↓ queue builder
asset_manifest.yaml + layer_map.yaml（派生物）
  ↓ PSD backend（現在はbuild planのみ）
model_import.psd
  ↓ live2d-cubism-workflow
Cubism import → auto mesh → parameter/API確認 → cubism_evaluation.yaml
  ↓ 素材起因の問題
asset_feedback.yaml → image-to-live2d-assetsの対象jobへ戻す
```

責務は次のように分けます。

| Skill | 担当 | 担当しないこと |
|---|---|---|
| `character-spec-generator` | 画像観測、ユーザー希望、model scope、motion、runtimeの仕様化 | 素材生成、PSD、Cubism操作 |
| `image-to-live2d-assets` | 画像解析計画、canonical queue、mask、inpainting、prompt、派生manifest/layer map、PSD出力計画 | Cubism操作、リギング、完成モデル判定 |
| `live2d-cubism-workflow` | 承認済みPSDのimport、高レベルUI macro、External API、Cubism評価とfeedback変換 | 隠れ素材の生成、権利判断、自由座標操作 |
| `live2d-one-image-vtuber` | 3 Skillのhandoff、gate、feedback loop | 各工程の実処理、Agents SDK state machine |

現在のMVPは、review-firstの自動segmentation候補生成からsource抽出、生成inpainting、候補ranking、再合成、global quality、failed-part refinementまでをrun単位のorchestratorで接続しています。生成backendはmockとoptional Diffusers / FLUX Fill、segmentation backendはmockとoptional SAM 2 / Grounded SAM 2です。redrawと実PSD writerは未接続です。素材状態を変更する単一ソースは `asset_generation_queue.yaml` で、segmentation result、assignment、inpainting result、selection、manifest、mask manifest、layer mapはいずれも派生成果物としてcanonical queueを直接変更しません。`tools.psd_asset_builder` は空のPSDを作らず、実backend接続前は必ず `status: plan_only` を返します。

著作権、二次創作ガイドライン、肖像・商標、画像生成サービスの利用条件を含め、source画像をモデル化する権利が不明な場合は先へ進めません。`source_image.rights_status: confirmed` はユーザーが権利を確認した後だけ設定してください。

## セットアップ

Python 3.11以上を使います。Windowsで実機GUI操作も行う場合は `windows` extraを入れてください。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[windows,dev]"
python -m pytest -q
```

生成inpaintingの実model adapterを使う場合だけ `python -m pip install -e ".[inpainting]"` を追加します。
checkpointは暗黙に取得せず、`local_files_only: true` が既定です。詳しい安全境界、prompt、候補生成、
ranking、レビュー、queue更新候補は [docs/inpainting.md](docs/inpainting.md) を参照してください。

### AI素材生成orchestrator

既定はdry-runで、ファイル作成もmodel loadも行いません。実行時はrunごとのディレクトリへ成果物を分離し、assignmentとselectionのreview gateで停止します。deterministicなmock CIだけは明示的な `--auto-approve-mock` で自動続行できます。

```powershell
python -m tools.asset_generation_orchestrator examples/asset_generation_queue.sample.yaml --segmentation-backend mock --inpainting-backend mock
python -m tools.asset_generation_orchestrator generated/asset_generation_queue.yaml --segmentation-backend mock --inpainting-backend mock --run-id run-001 --auto-approve-mock --execute
python -m tools.asset_generation_orchestrator generated/asset_generation_queue.yaml --segmentation-backend sam2 --inpainting-backend disabled --run-id segment-001 --execute
python -m tools.asset_generation_orchestrator generated/asset_generation_queue.segmented.yaml --segmentation-backend disabled --inpainting-backend diffusers --run-id inpaint-001 --execute
```

中断runは同じ入力とrun IDに `--resume` を付けて再開します。canonical queueまたはsource画像が変わったrunはstaleとして拒否されます。詳しい成果物、review手順、安全境界は [docs/asset-generation-orchestration.md](docs/asset-generation-orchestration.md) を参照してください。
FLUX Fillを含むmodel IDはhard-codeしていません。使用するcheckpointごとに配布元のライセンス、商用利用、
出力利用、アクセス条件を確認してください。adapterやoptional extraの導入はmodel利用許諾を意味しません。

通常のpytestはwall-clock benchmarkを除外します。2048x2048性能計測を明示的に実行するときだけ次を使います。

```powershell
python -m pytest -m benchmark
python -m pytest -m gpu
```

## no-assets dry-run

source画像やPSDがなくても、次のaction planはvalidatorとBridge dry-runを通せます。`--execute` は付けません。

```powershell
python -m tools.cubism_ui save
python -m tools.cubism_ui apply-auto-mesh --preset Standard --alpha 10
python -m tools.cubism_api get-documents
python -m scripts.validate_action_plan examples/action_plan.no_assets.sample.yaml
python -m tools.cubism_bridge run-action-plan examples/action_plan.no_assets.sample.yaml
```

## 自動segmentation

既定はdry-runです。dry-runはqueueと要求を検証しますが、source画像、fixture、SAM 2、Grounding DINOをloadせず、成果物も書きません。mock backendを実際に書き出す場合も`--execute`が必要です。

```powershell
python -m tools.automatic_segmenter generated/asset_generation_queue.yaml --backend mock --output generated/segmentation_result.yaml
python -m tools.automatic_segmenter generated/asset_generation_queue.yaml --backend mock --output generated/segmentation_result.yaml --execute
python -m tools.segmentation_candidate_ranker generated/segmentation_result.yaml --output generated/segmentation_ranked.yaml --execute
python -m tools.segmentation_assignment_planner generated/asset_generation_queue.yaml generated/segmentation_ranked.yaml --output generated/segmentation_assignment.yaml --execute
```

assignment planは常に`review_status: pending`、各assignmentは`status: needs_review`で生成されます。レビュー後にrootを`approved`、採用partを`status: approved`かつ`requires_review: false`へ変更したplanだけを、新しいqueue候補へ適用できます。入力queueは上書きできません。

```powershell
python -m tools.segmentation_assignment_planner apply generated/asset_generation_queue.yaml generated/segmentation_assignment.yaml --output generated/asset_generation_queue.segmented.yaml --execute
```

SAM 2はoptional dependencyです。`.[segmentation]`は補助runtimeを入れますが、Meta SAM 2自体は公式手順で別途導入してください。checkpoint利用時は`--checkpoint`と、導入済みSAM 2 package内のHydra config名を`--model-config`へ指定します。`--model-id`は公式SAM 2が対応するHugging Face IDのcache済みcheckpointだけを`local_files_only`で解決し、network downloadは開始しません。Grounded SAM 2はlocal Grounding DINO directoryを`--grounding-model`で明示します。未導入・未cache・設定不足は`unavailable`として停止します。

```powershell
python -m tools.automatic_segmenter generated/asset_generation_queue.yaml --backend sam2 --checkpoint C:\models\sam2.pt --model-config configs/sam2.1/sam2.1_hiera_l.yaml --device cuda:0 --output generated/segmentation_result.yaml --execute
python -m tools.automatic_segmenter generated/asset_generation_queue.yaml --backend grounded-sam2 --checkpoint C:\models\sam2.pt --model-config configs/sam2.1/sam2.1_hiera_l.yaml --grounding-model C:\models\grounding-dino --device cuda:0 --output generated/segmentation_result.yaml --execute
```

詳細な成果物契約、ranking観点、レビュー手順、実装済み／未実装境界は[`docs/segmentation.md`](docs/segmentation.md)を参照してください。実modelテストは通常CIから分離し、`python -m pytest -m gpu`でのみ実行します。

## real-assets workflow

以下は実在する `assets/source/character.png` が必要です。source画像はGit管理されません。

```powershell
python -m tools.material_planner assets/source/character.png --model-scope bust_up --motion-level standard
python -m tools.asset_feedback_validator examples/asset_feedback.sample.yaml --layer-map examples/layer_map.sample.yaml
python -m tools.asset_generation_queue_validator examples/asset_generation_queue.sample.yaml
python -m tools.asset_queue_builder examples/asset_generation_queue.sample.yaml
python -m tools.asset_manifest_validator examples/asset_manifest.sample.yaml --base-dir .
python -m tools.psd_asset_builder examples/asset_manifest.sample.yaml --base-dir .
```

上のqueueコマンドはサンプルの構造確認用で、未解決feedbackがあるため `merge_ready: false` が正常です。builderも既定はdry-runです。実成果物を書き出すときだけ `--execute` を付けます。

checked-in sample queueはvalidatorとhandoff契約を示すコンパクトなfixtureで、taxonomyの全partを含む完成モデル仕様ではありません。実作業ではplanner出力から `generated/asset_generation_queue.yaml` を作り、trackedされた `examples/*.sample.yaml` を `--force` で作業ファイルとして上書きしないでください。

```powershell
python -m tools.asset_queue_builder generated/asset_generation_queue.yaml --base-dir . --execute --force
```

### 実素材生成・再合成・品質gate

次のコマンドはdry-runではファイルを変更しません。`--execute` を付ける場合、source PNG、各mask PNG、抽出済みpart PNGがqueue/mask manifestの参照先に実在する必要があります。mask candidate generatorが書くのはmaskの計画manifestであり、意味segmentationによるmask画像そのものではありません。

```powershell
python -m tools.mask_candidate_generator generated/asset_generation_queue.yaml --base-dir .
python -m tools.mask_candidate_generator generated/asset_generation_queue.yaml --base-dir . --execute --force
python -m tools.part_extractor generated/mask_manifest.yaml --part eye_white_L --base-dir . --execute --force
python -m tools.hidden_region_completer generated/mask_manifest.yaml --part hair_back --base-dir . --execute --force
python -m tools.asset_recomposer generated/mask_manifest.yaml --base-dir . --output generated/reconstructed_source.png --difference-output generated/source_difference.png --execute --force
python -m tools.asset_quality_evaluator generated/mask_manifest.yaml --base-dir . --reconstructed generated/reconstructed_source.png --difference-output generated/quality_difference.png --output generated/asset_quality.yaml --execute --force
python -m tools.motion_stress_tester generated/mask_manifest.yaml --part hair_back --distance 8 --base-dir . --output generated/motion_hair_back.png --execute --force
python -m tools.asset_refinement_planner generated/asset_generation_queue.yaml generated/asset_quality.yaml --output generated/refinement_plan.yaml --refined-queue-output generated/asset_generation_queue.refined.yaml --execute --force
```

`hidden_region_completer` 単体のローカルbackendが実行できるのは `extract_and_edge_repair` と `transparency_fill` だけです。前者は `edge_extension_mask` のsource RGBAを追加してからtarget edge近傍のRGB fringeを補修し、後者はbinary `inpaint_mask` 内の完全透明穴だけを補完します。生成 `inpaint` は `asset_generation_orchestrator` からgenerative backendへ接続し、`redraw` は未接続エラーで停止します。refinementはpreserve違反をsource再抽出、edge-extension差分をedge repair再実行、inpaint mask外差分を同じinpaint方式のmask compositing修正、境界不連続を候補再生成またはrankingへ振り分けます。

mask manifestとasset quality reportはschema v2です。quality contractは、protect領域の完全一致を `preserve_region_difference_score`（閾値0固定）、source-preserving補修を `edge_extension_difference_score`（小さい設定可能閾値）、生成可能領域のsource差分を `inpaint_region_source_difference_score`（記録専用）、再合成結果を `visual_reconstruction_difference_score`（foreground/reconstruction mask内）として分離します。inpaint source差分は、source上の前面髪と生成した額のように正しい補完でも大きくなるためPASS/FAILには使いません。

inpaint gateはprotect領域、inpaint mask外の変更、canvas/origin、target必須coverageのalpha hole、edge continuity、boundary color、宣言mask外への漏れを検査します。`inpaint_mask` は変更許可領域であり、その全域を不透明に埋めることは要求しません。境界metricは実際に生成されたinpaint画素が既存target/protect/edge-extensionへ接するseamだけを評価し、対応する8近傍ペアの絶対差を平均します。standaloneの隠れpartの外周透明は不連続として扱いません。`max_inpaint_outside_difference_score` は0固定です。`max_edge_extension_difference_score`、`max_edge_continuity_score`、`max_boundary_color_difference_score`、`max_visual_reconstruction_difference_score` はCLIとquality YAMLで調整できますが、inpaint source差分に対応する閾値はありません。quality、recomposer、motion、refinementは同じ `include_in_import: true` layer集合を使い、guide/mask assetを除外します。`overlap_margin_px` だけでtargetを自動膨張せず、desired coverageはtargetと `edge_extension_mask` だけから導出します。

差分はRGBAをpremultiplied alphaへ変換してから比較するため、完全透明ピクセルのRGBだけが違ってもfailureになりません。sourceに背景が含まれていてもモデル素材へ背景を含めない場合は、全import partのtarget/edge/inpaint領域とreconstruction alphaの和集合をforeground/reconstruction maskとし、その範囲だけをglobal評価します。reconstruction alphaを含めることで、宣言mask外へはみ出した不透明画素も検出します。各partにも同じ再合成差分を担当領域で計測し、global failureを可能な限りfailed partへ帰属させます。

motion stress previewは、全import partを再合成し、指定partだけを左右へ動かした3フレームの非gate目視資料です。part単体が必要な場合だけ `--debug-part-only` を使います。preview単独でPASSやCubism deformation品質を主張しません。`draw_order` は小さい値が背面、大きい値が前面です。

maskのbinary判定は既定でalpha 1以上を対象にしますが、共通APIの `load_binary_mask(..., alpha_threshold=...)` で用途別thresholdを指定できます。PNG出力は同一ディレクトリの一時ファイルへ保存後、atomic replaceします。透明穴fillはinpaint maskのbounding box外を走査しません。

`asset_quality_evaluator` のdry-runは画像を読まず `quality_result: not_run`、`evaluated_parts: 0` を返します。品質PASS/FAILは `--execute` で実PNGを評価し、全import partを含むquality YAMLが生成された場合だけ使用します。

実素材をCubismへhandoffする直前は、生成済みmanifestをcanonical queueと照合して必須gateを実行します。

```powershell
python -m tools.asset_generation_queue_validator generated/asset_generation_queue.yaml --base-dir . --manifest generated/asset_manifest.yaml --require-merge-ready
python -m tools.asset_manifest_validator generated/asset_manifest.yaml --base-dir . --require-handoff-ready
```

1つ目のコマンドはqueueの `derivatives.asset_manifest` と指定manifestの実パス・全canonical内容を照合し、2つ目はmanifest本体と実ファイルのCubism handoff readinessを検証します。どちらか片方だけではhandoff gate完了と扱いません。

### validation mode

`asset_generation_queue.yaml` と `cubism_evaluation.yaml` は次のmodeを持ちます。

- `strict`: 現行のhandoff向け。required checkのWARNや矛盾したapproved状態をエラーにする。構造が正しいWARN/FAILは修正用feedbackへ変換できる。
- `dev`: 開発途中のWARNを許容する。ただし構造エラー、FAIL、project/layer不一致は許容せず、WARNがある状態をhandoff readyとは扱わない。

現行の素材queue schemaはv3です。v2 queueはvalidatorとmanifest/layer map builderで読み取り可能ですが、target/protect/edge-extension/inpaintの四mask、quality、refinement fieldがないため、実素材生成pipelineへ進む前にv3 templateへ移行してください。

計画YAMLだけを書き出す場合は `--execute`、PSD build planを書き出す場合は `--write-plan` を使います。後者もPSD自体は生成しません。

```powershell
python -m tools.material_planner assets/source/character.png --model-scope half_body --motion-level expressive --execute --output outputs/asset_generation_plan.yaml
python -m tools.psd_asset_builder examples/asset_manifest.sample.yaml --base-dir . --write-plan outputs/psd_build_plan.yaml
```

次のコマンドは実在する `generated/model_import.psd` と `generated/layer_map.yaml` を要求します。サンプルYAMLだけでは実行できません。

```powershell
python -m scripts.validate_action_plan examples/action_plan.real_assets.sample.yaml
python -m tools.cubism_ui import-psd generated/model_import.psd
python -m tools.cubism_bridge run-action-plan examples/action_plan.real_assets.sample.yaml
```

## 実機Cubismで実行する

1. Live2D Cubism Editorを起動し、Modeling Workspaceを表示します。
2. External Application Integrationを有効化します。初回接続ではCubism側の「Allow」が必要です。
3. 先にdry-run出力を確認します。
4. 問題なければ `--execute` を付けます。

```powershell
python -m tools.cubism_api --execute register
python -m tools.cubism_api --execute get-approval
python -m tools.cubism_ui import-psd generated/model_import.psd --execute
python -m tools.cubism_bridge apply-auto-mesh-and-capture --preset Standard --alpha 10 --execute
```

操作後は評価を記録し、素材起因のWARN/FAILだけをfeedbackへ変換します。

```powershell
python -m tools.cubism_evaluation validate examples/cubism_evaluation.sample.yaml --layer-map examples/layer_map.sample.yaml
python -m tools.cubism_evaluation to-feedback examples/cubism_evaluation.sample.yaml --layer-map examples/layer_map.sample.yaml --output outputs/asset_feedback.yaml
```

Cubism APIのトークンは `.live2d-agent/cubism-token.json` に保存され、Git管理から除外されます。`SetParameterValues` はEditorの一時バッファへ値を送ります。解除には `clear-parameter-values` を使います。

## action plan

```powershell
python -m scripts.validate_action_plan examples/action_plan.no_assets.sample.yaml
python -m scripts.validate_action_plan examples/action_plan.real_assets.sample.yaml
python -m tools.cubism_bridge run-action-plan examples/action_plan.no_assets.sample.yaml
python -m tools.cubism_bridge run-action-plan examples/action_plan.real_assets.sample.yaml --execute
```

`action_plan.no_assets.sample.yaml` はAPI/UI payloadのdry-run境界確認用です。`action_plan.real_assets.sample.yaml` は承認済みPSDとlayer mapを使う実機用で、素材なしでは実行しません。

queue validatorは `feedback_inputs` の実YAMLとlayer mapも読み、queue・feedback・layer mapのproject、feedback IDのjob割当、severityに関係なく未解決のissueをmerge gateへ反映します。

`mode` は次の4種類です。

- `file`: 仕様・レイヤーマップなどの決定的検証
- `ui_macro`: 許可済みのCubism高レベルUIマクロ
- `api`: 許可済みのCubism External API操作
- `manual_checkpoint`: 見た目判断や想定外状態の確認。操作の肩代わりには使わない

未知のコマンド、座標指定、`click` / `drag` 系コマンドはvalidatorが拒否します。実行は最初の失敗で停止し、`outputs/action_plan_report.md` に途中結果を残します。
コマンドごとの必須・許可引数も検証し、action planからのスクリーンショット出力先は `outputs/*.png` に限定します。

## ディレクトリ

```text
.agents/skills/character-spec-generator/  source画像からcharacter specを作るSkill
.agents/skills/image-to-live2d-assets/   1枚絵からの素材生成Skillとテンプレート
.agents/skills/live2d-cubism-workflow/  承認済み素材をCubismで操作するSkill
.agents/skills/live2d-one-image-vtuber/  全体handoffとfeedback loop
tools/material_planner.py                scope/level別の素材計画
tools/asset_generation_queue_validator.py 並列queueとmerge gateの検証
tools/asset_queue_builder.py              queueからmanifest/layer mapを生成
tools/mask_candidate_generator.py         queueからmask manifestを生成（mask画像backendは未接続）
tools/part_extractor.py                   target maskによるsource RGBA抽出
tools/hidden_region_completer.py          source保護付きの限定的な透明領域補完
tools/asset_recomposer.py                 draw order再合成とsource差分
tools/asset_quality_evaluator.py          halo・穴・overlap・保護画素差分の簡易品質gate
tools/motion_stress_tester.py             全import part再合成の平行移動preview
tools/asset_refinement_planner.py         failed partだけの再生成計画
tools/asset_feedback_validator.py        Cubism feedbackとlayer IDの検証
tools/cubism_evaluation.py               Cubism評価とfeedback変換
tools/asset_manifest_validator.py        manifestとhandoff gateの検証
tools/psd_asset_builder.py               PSD build plan stub
tools/cubism_ui.py                      Windows UIマクロ
tools/cubism_api.py                     WebSocket External API
tools/cubism_bridge.py                  action plan実行とレポート
scripts/                                構造成果物validator
examples/                               学習用の仕様・レイヤー・計画
schemas/                                mask manifest・asset quality schema
assets/models/                          ユーザー所有PSD置き場（Git対象外）
assets/source/                          ユーザー所有source画像（Git対象外）
generated/                              生成part・manifest・PSD（Git対象外）
outputs/                                スクリーンショット・レポート（Git対象外）
.github/workflows/pytest.yml            push/PR時のpytest
```

## 安全性と既知の制約

- `image-to-live2d-assets` はsource画像の利用権を自動判定しません。未確認ならhandoffを停止します。
- manifestとlayer mapはqueueの派生物です。派生物だけを修正しても次回生成で上書きされます。
- queue builderは派生出力を`--base-dir`内に限定し、manifest/layer mapを事前検査・一時書込み・rollback付きの1組として更新します。
- inferred素材は、queueの目視承認状態を更新してmanifestを再生成するまでCubism import可能と扱いません。
- handoff gateはsource画像・import PNG・PSD・layer mapの実在と基本signature、project/canvas/layer ID/nameの一致も確認します。画像の完全decodeやPSD内部構造検証は今後のbackend検証対象です。
- `required: true` の部品がimport対象外、またはredraw素材が未レビューならhandoffしません。
- 意味segmentation、生成inpainting、review gate、再合成、quality、refinementは統合orchestratorへ接続済みです。実modelはoptional依存とローカルmodelを明示した場合だけ動作します。redraw、Photoshop Plugin、実PSD writerは未接続です。
- part attribution領域が重なる場合、同じ視覚差分が複数partへ帰属する可能性があります。実画像でrefinement過多を観測し、必要ならownership maskを追加します。
- binary alpha判定値1とwhite-halo RGB判定値245は現在の固定実装値で、quality artifactの可変thresholdにはまだ含めていません。
- 実機操作はWindows UI Automationのラベルを使います。Cubismの言語・版によってアクセシビリティ名が異なる場合、推測して続行せず停止します。
- 対象ウィンドウはタイトルだけでなく、実行ファイル名が既定の `CubismEditor*.exe` に一致することも確認します。ViewerやUpdaterは対象外で、ダイアログは特定済みEditor process IDに限定します。
- 現在のMVPは日本語/英語の既知ラベルを扱います。実機ごとのUIA調整は、失敗スクリーンショットとcontrol情報を見て名前付きlocatorを追加してください。
- 自動メッシュ生成はパラメータ設定後に行うと変形を戻す可能性があります。取り込み直後に行う前提です。
- auto-meshのUndo recoveryは、マクロがauto-mesh適用を完了したと記録した場合だけ実行します。結果は常に目視確認が必要です。
- `save` はSave Asダイアログが開いた場合に完了扱いせず停止します。新規モデルは保存先を明示した別の名前付きマクロを追加してから自動保存対象にしてください。
- 実装時のローカル環境ではCubismの起動・標準インストール先を確認できなかったため、実機UIの成功までは保証していません。dry-runとモック境界は検証対象です。
- Cubism/VTube Studioで目視確認するまで、モデル完成・本番投入可能とは扱いません。

## 次の拡張

CLIの入出力が実機で安定した後に進めます。

1. segmentation / 生成inpainting実modelのGPU fixtureとredraw / 外部画像生成tool adapter
2. Photoshop PluginまたはPSD writer backend
3. 成熟したvalidatorとBridgeコマンドをMCP toolとして公開
4. UIA control treeの診断コマンドと版別profile
5. PSDレイヤー解析とparameter CSV生成
6. Director / Asset Evaluator / Layer Designer / Cubism PlannerへのAgent分割
7. Cubism書き出し通知とVTube Studio QA

## 参照した公式仕様

- [Live2D Cubism Editor External API](https://docs.live2d.com/en/cubism-editor-manual/external-application-integration-api/)
- [External API integration functions](https://docs.live2d.com/en/cubism-editor-manual/external-application-integration-api-list/)
- [PSD import](https://docs.live2d.com/en/cubism-editor-manual/psd-import/)
- [Automatic Mesh generator](https://docs.live2d.com/cubism-editor-manual/mesh-edit/)
