#!/usr/bin/env bash

LOG_FILE="./db_mem_usage.log"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# 要监控的数据库进程关键字
DB_PROCESSES=(
  "redis-server"
  "mongod"
  "elasticsearch"
  "milvus"
)

echo "==============================" >> "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" >> "$LOG_FILE"

for DB in "${DB_PROCESSES[@]}"; do
  echo "[$DB]" >> "$LOG_FILE"

  # 找 PID（排除 grep 自己）
  PIDS=$(ps aux | grep "$DB" | grep -v grep | awk '{print $2}')

  if [ -z "$PIDS" ]; then
    echo "  Not running" >> "$LOG_FILE"
    continue
  fi

  TOTAL_RSS=0

  for PID in $PIDS; do
    # RSS 单位：KB
    RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null)

    if [ -n "$RSS_KB" ]; then
      TOTAL_RSS=$((TOTAL_RSS + RSS_KB))
      echo "  PID $PID RSS: $((RSS_KB / 1024)) MB" >> "$LOG_FILE"
    fi
  done

  echo "  Total RSS: $((TOTAL_RSS / 1024)) MB" >> "$LOG_FILE"
done
