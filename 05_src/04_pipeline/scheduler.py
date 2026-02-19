#!/usr/bin/env python3
"""
파이프라인 cron 스케줄러

config.yaml의 cron 표현식을 읽어 crontab에 등록/해제한다.

사용법:
  python scheduler.py --enable    # cron 등록
  python scheduler.py --disable   # cron 해제
  python scheduler.py --status    # 현재 상태 확인
"""
import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

CONFIG_PATH = Path(__file__).parent / "config.yaml"
PIPELINE_SCRIPT = Path(__file__).parent / "run_pipeline.py"
CRON_MARKER = "# whypi-pipeline-auto"


def load_cron_expr() -> str:
    """config.yaml에서 cron 표현식 로드"""
    if yaml is None:
        return "0 3 1 * *"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("pipeline", {}).get("auto_schedule", {}).get("cron", "0 3 1 * *")


def get_current_crontab() -> str:
    """현재 crontab 내용 반환"""
    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return ""


def set_crontab(content: str):
    """crontab 내용 설정"""
    process = subprocess.Popen(
        ["crontab", "-"], stdin=subprocess.PIPE, text=True
    )
    process.communicate(input=content)


def enable():
    """cron 등록"""
    cron_expr = load_cron_expr()
    python_path = sys.executable
    script_path = str(PIPELINE_SCRIPT.resolve())

    cron_line = f"{cron_expr} {python_path} {script_path} --local-only {CRON_MARKER}"

    current = get_current_crontab()

    # 기존 항목 제거
    lines = [line for line in current.splitlines() if CRON_MARKER not in line]
    lines.append(cron_line)

    set_crontab("\n".join(lines) + "\n")
    print(f"cron 등록 완료: {cron_line}")


def disable():
    """cron 해제"""
    current = get_current_crontab()
    lines = [line for line in current.splitlines() if CRON_MARKER not in line]
    set_crontab("\n".join(lines) + "\n")
    print("cron 해제 완료")


def status():
    """현재 상태 확인"""
    current = get_current_crontab()
    found = [line for line in current.splitlines() if CRON_MARKER in line]

    if found:
        print("상태: 활성")
        for line in found:
            print(f"  {line}")
    else:
        print("상태: 비활성 (cron 미등록)")

    # config.yaml 설정도 표시
    schedule_cfg = {}
    if yaml is not None and CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        schedule_cfg = config.get("pipeline", {}).get("auto_schedule", {})
    print(f"\nconfig.yaml 설정:")
    print(f"  enabled: {schedule_cfg.get('enabled', False)}")
    print(f"  cron: {schedule_cfg.get('cron', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="파이프라인 cron 스케줄러")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enable", action="store_true", help="cron 등록")
    group.add_argument("--disable", action="store_true", help="cron 해제")
    group.add_argument("--status", action="store_true", help="현재 상태 확인")
    args = parser.parse_args()

    if args.enable:
        enable()
    elif args.disable:
        disable()
    elif args.status:
        status()


if __name__ == "__main__":
    main()
