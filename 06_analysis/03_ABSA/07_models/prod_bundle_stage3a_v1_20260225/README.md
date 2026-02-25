# Stage 3A 운영 번들 (prod_bundle_stage3a_v1_20260225)

모델 재학습 없이 후처리만 변경한 Stage 3A 최종 운영 세팅.
이 폴더의 4개 파일이 "한 세트"이며, 개별 수정 금지.

## 파일 목록

| 파일 | 설명 |
|------|------|
| `best_model_stage2_v1.pt` | Stage 2 학습 모델 (KcELECTRA-base, 8 aspects) |
| `none_thresholds_stage3a.json` | Per-aspect none threshold + polar_threshold=0.50 + design_rule=true |
| `design_rule_config.json` | Design Rule Override 키워드 (Tier1~3) + Keyword Gate |
| `golden_eval_metrics_test.json` | Golden Test 평가 결과 (GO 판정 증빙) |
| `MANIFEST.json` | 버전 정보, MD5 체크섬, 모든 설정값 기록 |

## 사용법

```python
from RQ_absa.s8_inference import ABSAInference

# 번들에서 한 줄로 로드 (무결성 검증 포함)
inference = ABSAInference.from_bundle(
    "07_models/prod_bundle_stage3a_v1_20260225"
)

# 추론 실행
results = inference.infer_dataframe(df, text_column="text")
```

## 무결성 검증

```bash
cd 07_models/prod_bundle_stage3a_v1_20260225

# MANIFEST.json의 MD5와 실제 파일 비교
python3 -c "
import hashlib, pathlib, json
m = json.load(open('MANIFEST.json'))
for f, info in m['files'].items():
    if 'md5' not in info:
        continue
    actual = hashlib.md5(pathlib.Path(f).read_bytes()).hexdigest()
    status = 'OK' if actual == info['md5'] else 'MISMATCH'
    print(f'{f}: {status}')
"
```

## 핵심 설정값

| 설정 | 값 | 의미 |
|------|---|------|
| 사용감/성능 threshold | 0.03 | 검출 매우 억제 (낮을수록 검출 줄어듦) |
| 디자인 threshold | 0.50 | Design Rule이 대체하므로 사실상 무의미 |
| polar_threshold | 0.50 | max(P(pos), P(neg)) < 0.50 → neutral |
| design_rule | true | 디자인 aspect를 키워드 규칙으로 판단 |

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1 | 2026-02-25 | 최초 번들 생성 (Stage 3A GO 판정 후) |
