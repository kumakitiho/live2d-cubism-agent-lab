# Cubism External API Policy

- 既定接続先は `ws://127.0.0.1:22033`。
- 接続ごとに `RegisterPlugin` を実行し、保存済みtokenを渡す。
- 初回はCubism側のAllowが必要。未承認のまま他命令を送らない。
- Request/Responseを1つずつ完了させ、RequestIdを照合する。
- tokenは `.live2d-agent/cubism-token.json` に保存し、Gitへ含めない。
- `SetParameterValues` 後は必要に応じて `ClearParameterValues` を実行する。
- UIDは接続で変わり得るため永続IDとして扱わない。
- import前後を比較するBridge実行では1本のWebSocket sessionを維持し、同一session内のDocumentUID差分だけを使う。

利用するversion:

- `GetIsApproval`: 0.9.0
- `GetDocuments`: 0.9.1
- `GetCurrentModelUID`: 0.9.1
- `GetCurrentEditMode`: 0.9.0
- `GetParameters`: 1.0.1
- `GetParameterValues` / `SetParameterValues` / `ClearParameterValues`: 0.9.1
- `SendCubismLog`: 0.9.3
