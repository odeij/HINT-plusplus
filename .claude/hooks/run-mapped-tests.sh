#!/bin/bash
# PostToolUse hook: run the pytest target mapped to the edited file.
# stdin: hook JSON; relevant field tool_input.file_path (Edit/Write).
# Exit 0 = silent pass-through; exit 2 = blocking feedback (stderr shown to Claude).

set -u
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
[ -z "$FILE_PATH" ] && exit 0

REPO="${CLAUDE_PROJECT_DIR:-/home/ahmad/Desktop/HINT++}"
PY=/home/ahmad/anaconda3/bin/python
cd "$REPO" || exit 0

REL="${FILE_PATH#"$REPO"/}"
TARGET=""
case "$REL" in
  src/safety/*|src/memory/*|src/adaptation/*|harness/*)
    base=$(basename "$REL" .py)
    if [ -f "tests/test_${base}.py" ]; then
      TARGET="tests/test_${base}.py"
    else
      TARGET="tests/"
    fi
    ;;
  tests/test_*.py)
    TARGET="$REL"
    ;;
  tests/*)
    TARGET="tests/"   # conftest.py or nested test files affect the whole suite
    ;;
  *)
    exit 0
    ;;
esac

OUT=$(PYTHONPATH=. "$PY" -m pytest "$TARGET" -q --tb=line 2>&1)
STATUS=$?
if [ $STATUS -ne 0 ]; then
  {
    echo "Mapped pytest target FAILED after editing $REL (target: $TARGET):"
    echo "$OUT" | grep -E "FAILED|ERROR|error|assert" | head -20
    echo "$OUT" | tail -3
  } >&2
  exit 2
fi
exit 0
