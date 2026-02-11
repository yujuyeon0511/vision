# NLP Research Workflow

멀티모달 NLP (Vision-Language) 연구를 위한 구조화된 워크플로우 시스템.
Claude Code에서 `/research` 커맨드로 연구의 전 과정을 관리한다.

---

## 시작하기

### 사전 준비

- [Claude Code CLI](https://claude.ai/claude-code)가 설치되어 있어야 한다.

### 실행 방법

```bash
cd ~/research
claude
```

`~/research/` 디렉토리에서 Claude Code를 실행하면, `.claude/` 안의 설정(에이전트, 스킬, 템플릿)이 자동으로 로드된다.

---

## 연구 라이프사이클

연구는 5단계 순환 구조로 진행된다. 각 단계를 순서대로 밟되, 결과에 따라 1단계로 돌아가 반복할 수 있다.

```
1. Hypothesis  →  2. Experiment Design  →  3. Experiment Run
                                                    ↓
         5. Paper Writing  ←  4. Result Analysis ←──┘
                ↓
           (완료 or 1단계로 돌아가 반복)
```

| 단계 | 활동 | 설명 |
|------|------|------|
| 1 | Hypothesis | 연구 질문 정의, 가설 수립, 문헌 조사 |
| 2 | Experiment Design | 모델 구조, 데이터셋, 하이퍼파라미터, 베이스라인 설계 |
| 3 | Experiment Run | 실험 실행, 환경 스냅샷 기록, GPU 로그 |
| 4 | Result Analysis | 결과 비교, 통계 검증, 에러 분석, 가설 검증 |
| 5 | Paper Writing / Iterate | 논문 섹션 작성 또는 다음 실험 개선안 도출 |

---

## 커맨드 사용법

Claude Code 안에서 `/research` 뒤에 서브커맨드를 입력한다.

### 1단계: 가설 수립

```
/research hypothesis vl-alignment
```

- `vl-alignment` 주제에 대한 가설 문서를 생성한다.
- `literature-reviewer` 에이전트가 관련 논문을 검색하고 요약한다.
- 연구 질문(RQ), 가설(H1/H0), 문헌 리뷰, 실현가능성 평가를 포함한 문서가 만들어진다.
- 생성 위치: `docs/01-hypothesis/features/vl-alignment-hypothesis.md`

### 2단계: 실험 설계

```
/research design vl-alignment
```

- 1단계에서 작성한 가설 문서를 기반으로 실험을 설계한다.
- `experiment-designer` 에이전트가 다음을 수행한다:
  - 실험 ID 자동 생성 (예: `EXP-20260211-001`)
  - 모델 구조, 데이터셋, 하이퍼파라미터, 베이스라인, 어블레이션 계획 작성
  - 실험 config YAML 파일 생성
  - GPU 사용 가능 여부 및 소요 시간 추정
- 생성 위치:
  - 설계 문서: `docs/02-experiment-design/features/EXP-20260211-001-design.md`
  - Config: `experiments/configs/EXP-20260211-001-config.yaml`

### 3단계: 실험 실행

```
/research run vl-alignment
```

- 실험 실행을 위한 가이드와 환경 스냅샷을 생성한다.
- 자동으로 기록하는 정보:
  - GPU 정보 (`nvidia-smi`)
  - Python/PyTorch/CUDA 버전
  - git 상태 (branch, commit hash)
  - 호스트명, 디스크 용량
- 학습 커맨드, 모니터링 팁, 체크포인트 저장 가이드를 제공한다.
- 생성 위치: `docs/03-experiment-log/EXP-20260211-001-log.md`

### 4단계: 결과 분석

```
/research analyze vl-alignment
```

- `result-analyzer` 에이전트가 실험 결과를 분석한다:
  - 베이스라인 대비 성능 비교표 생성
  - 통계적 유의성 검증 (paired t-test, bootstrap)
  - 신뢰구간, 효과 크기(Cohen's d) 계산
  - 에러 카테고리 분류 및 실패 사례 분석
  - 시각화 스크립트 생성 (학습 곡선, 비교 차트 등)
  - 가설 지지 여부 판정
- 생성 위치:
  - 분석 문서: `docs/04-results/features/EXP-20260211-001-analysis.md`
  - 결과 테이블: `results/tables/`
  - 시각화: `results/figures/`

### 5단계: 논문 작성

```
/research paper vl-alignment
```

- `paper-writer` 에이전트가 이전 단계의 모든 문서를 읽고 논문 섹션을 작성한다.
- ACL/NeurIPS/EMNLP 스타일에 맞춰 다음 섹션을 생성한다:
  - Abstract, Introduction, Related Work, Method, Experiments, Discussion, Conclusion
- Markdown으로 작성하되, LaTeX 변환 힌트를 HTML 주석으로 포함한다.
- 생성 위치: `docs/05-paper/sections/` 아래 섹션별 파일

### 반복: 다음 실험

```
/research iterate vl-alignment
```

- 분석 결과를 기반으로 다음 실험 개선안을 도출한다.
- 새로운 가설 문서를 만들어 다음 사이클을 시작한다.
- 이전 실험 결과를 참조하며, 성공 기준을 업데이트한다.

### 상태 확인

```
/research status
```

- 현재 연구 진행 상태를 한눈에 보여준다.
- 출력 예시:
  ```
  === NLP Research Status ===
  Project: vl-alignment
  Current Phase: experiment_design
  Iteration: 1

  Phases:
    [x] Hypothesis      - completed (1 documents)
    [>] Exp. Design     - in_progress (1 documents)
    [ ] Exp. Run        - not_started (0 experiments)
    [ ] Result Analysis - not_started (0 documents)
    [ ] Paper Writing   - not_started (0 sections)
  ```

### 다음 단계 안내

```
/research next
```

- 현재 상태를 읽고 다음에 무엇을 해야 하는지 안내한다.
- 예: "가설이 완료되었습니다. `/research design vl-alignment`로 실험을 설계하세요."

---

## 에이전트

각 연구 단계에 전문화된 AI 에이전트가 배정된다.

| 에이전트 | 모델 | 역할 |
|----------|------|------|
| `literature-reviewer` | sonnet | 논문 검색 및 요약, 연구 갭 식별, 가설 수립 지원 |
| `experiment-designer` | sonnet | 실험 설계, config YAML 생성, GPU 가용성 체크 |
| `result-analyzer` | sonnet | 결과 분석, 통계 검증, 시각화 스크립트 생성 |
| `paper-writer` | opus | 논문 섹션 작성 (Markdown + LaTeX 힌트) |

에이전트는 `/research` 커맨드 실행 시 자동으로 호출된다. 직접 호출할 필요 없다.

---

## 디렉토리 구조

```
~/research/
├── .claude/                           # Claude Code 설정
│   ├── CLAUDE.md                      # 프로젝트 컨텍스트
│   ├── agents/                        # 에이전트 정의 (4개)
│   │   ├── literature-reviewer.md
│   │   ├── experiment-designer.md
│   │   ├── result-analyzer.md
│   │   └── paper-writer.md
│   ├── skills/nlp-research/
│   │   └── SKILL.md                   # /research 커맨드 정의
│   └── templates/                     # 문서 템플릿
│       ├── hypothesis.template.md
│       ├── experiment-design.template.md
│       ├── experiment-log.template.md
│       ├── results-analysis.template.md
│       ├── paper-draft.template.md
│       └── shared/
│           └── nlp-research-conventions.md
├── docs/                              # 연구 문서 (단계별)
│   ├── .research-status.json          # 연구 상태 추적 파일
│   ├── 01-hypothesis/features/        # 가설 문서
│   ├── 02-experiment-design/features/ # 실험 설계 문서
│   ├── 03-experiment-log/             # 실험 실행 로그
│   ├── 04-results/features/           # 결과 분석 문서
│   └── 05-paper/sections/             # 논문 섹션 초안
├── experiments/
│   ├── configs/                       # 실험 config YAML
│   └── scripts/                       # 학습/평가 스크립트
├── results/
│   ├── tables/                        # 결과 테이블 (CSV, Markdown)
│   └── figures/                       # 그래프, 시각화
├── paper/                             # LaTeX 논문 소스
├── data/                              # 데이터셋 (gitignore됨)
├── .gitignore
└── README.md
```

---

## 컨벤션

### 실험 ID

```
EXP-YYYYMMDD-NNN
```

- `YYYYMMDD`: 실험 생성 날짜
- `NNN`: 당일 순번 (001, 002, ...)
- 예시: `EXP-20260211-001`

### 메트릭 보고 규칙

- 여러 시드(기본: 42, 123, 456)로 실행 후 **mean +/- std**로 보고
- 소수점: 일반 메트릭 4자리, 퍼센트 2자리
- 통계적 유의성: p < 0.05 기준, paired t-test 또는 bootstrap 사용
- 테이블에서 **최고 결과는 볼드**, _차선은 밑줄_

### 커밋 메시지

```
[phase] 간단한 설명

예시:
[hypothesis] VL alignment 연구 가설 정의
[design] EXP-20260211-001 어블레이션 config 추가
[experiment] EXP-20260211-001 결과 로그
[analysis] 주요 결과 통계 검증 추가
[paper] introduction 섹션 초안 작성
```

---

## 사용 예시: 처음부터 끝까지

```bash
# 1. 프로젝트 디렉토리에서 Claude Code 시작
cd ~/research
claude

# 2. 현재 상태 확인
/research status

# 3. 가설 수립 (주제: vision-language alignment)
/research hypothesis vl-alignment

# 4. 실험 설계
/research design vl-alignment

# 5. 실험 실행 가이드 받기
/research run vl-alignment

# 6. (실제 학습 실행 후) 결과 분석
/research analyze vl-alignment

# 7. 논문 작성
/research paper vl-alignment

# 8. 결과가 부족하면 다음 반복
/research iterate vl-alignment
```
