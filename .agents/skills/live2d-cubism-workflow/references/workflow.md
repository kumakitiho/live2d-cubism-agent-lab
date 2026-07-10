# Cubism workflow

## Input gate

- `model_import.psd` が存在し、上流manifestで承認済みである。
- `layer_map.yaml` がPSDのlayer ID/name/canvasと一致する。
- `action_plan.yaml` がallowlist validatorを通る。
- source権利、隠れ領域、mask、merge gateは上流で解決済みである。

## Operation

1. no-assets planでdry-run境界を確認する。
2. real-assets planをvalidatorへ通す。
3. import前後のdocument snapshotを比較する。
4. auto meshやparameter probeを名前付き操作で実行する。
5. reportとevidenceを保存する。

## Evaluation

- UI自動化の失敗はoperation reportへ記録する。
- eye、mouth、mesh、textureを `cubism_evaluation.yaml` の必須カテゴリとして評価する。
- `strict` はrequired checkのWARNを評価失敗とし、`dev` はWARNを許容する。FAILは両modeで評価失敗とする。ただし構造が正しいWARN/FAILは、修正用feedbackへ変換できる。
- 素材の欠け、境界、隠れ塗り、分割、style、transparencyのwarn/failを `asset_feedback.yaml` へ変換する。
- evaluationとfeedbackはlayer mapの実IDと照合してから `image-to-live2d-assets` へ返す。

## Completion

- action planが最後まで実行または明示的に停止している。
- blocking feedbackが残っていない。
- Cubism/VTube Studioの目視QAと未自動化項目が記録されている。
