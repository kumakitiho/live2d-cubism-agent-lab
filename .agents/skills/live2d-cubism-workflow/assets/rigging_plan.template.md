# Rigging plan

## Parameters

| Parameter ID | Range | Parts / deformers | Notes |
|---|---:|---|---|
| ParamAngleX | -30..30 | face hierarchy | head turn |
| ParamEyeLOpen | 0..1 | left eye | blink |
| ParamEyeROpen | 0..1 | right eye | blink |
| ParamMouthOpenY | 0..1 | mouth | lip sync |

## Deformer hierarchy

Describe face, eyes, mouth, hair, body, clothes, and accessory parent-child relationships.

## Expressions

List normal, smile, surprise, anger/sadness, and custom toggle parameters.

## Physics

List hair and accessories with input parameter, output parameter, root, and expected damping.

## QA

- extreme parameter combinations
- blink and eye tracking
- mouth open / form combinations
- head XYZ and body follow
- physics stability
- VTube Studio import and tracking
